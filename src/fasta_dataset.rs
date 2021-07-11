use std::cmp::{max, min};
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek};
use std::path::Path;
use std::sync::Arc;

use clam::dataset::RowMajor;
use clam::prelude::*;
use clam::CompressibleDataset;
use ndarray::prelude::*;
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FastaRecord {
    name: String,
    len: u64,
    offset: u64,
    line_bases: u64,
    line_bytes: u64,
}

pub struct FastaDataset {
    path: OsString,
    records: Vec<FastaRecord>,
    num_sequences: usize,
    max_seq_len: usize,
    metric: Arc<dyn Metric<u8, u64>>,
    batch_size: usize,
}

impl std::fmt::Debug for FastaDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("RowMajor Dataset")
            .field("num_sequences", &self.num_sequences)
            .field("max_seq_len", &self.max_seq_len)
            .field("metric", &self.metric.name())
            .finish()
    }
}

impl FastaDataset {
    pub fn new(fasta_path: &Path) -> std::result::Result<FastaDataset, String> {
        let mut fai_path = fasta_path.as_os_str().to_owned();
        fai_path.push(".fai");

        let fai = File::open(fai_path.clone()).expect(&format!("Could not open fai index for {:?}", fai_path)[..]);

        let records: Vec<FastaRecord> = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .from_reader(fai)
            .deserialize()
            .map(|row| row.expect("failed to read a row"))
            .collect();

        let num_sequences = records.len();
        let max_seq_len = records.iter().map(|f| f.len).max().unwrap() as usize;
        let batch_size = (num_sequences as f64).sqrt() as usize;
        let batch_size = max(batch_size, 2_500);

        Ok(FastaDataset {
            path: fasta_path.as_os_str().to_owned(),
            records,
            num_sequences,
            max_seq_len,
            metric: Arc::new(clam::metric::Hamming),
            batch_size,
        })
    }

    pub fn as_arc_dataset(self: Arc<Self>) -> Arc<dyn Dataset<u8, u64>> {
        self
    }

    // TODO: Merge these next two methods and use the permute crate
    pub fn subsample_indices(&self, subsample_size: usize) -> Vec<Index> {
        (0..self.num_sequences).choose_multiple(&mut rand::thread_rng(), subsample_size)
    }

    pub fn get_complement_indices(&self, indices: &[Index]) -> Vec<Index> {
        (0..self.num_sequences).filter(|i| !indices.contains(i)).collect()
    }

    pub fn get_subset_from_indices(&self, indices: &[Index]) -> Arc<RowMajor<u8, u64>> {
        let sequences: Vec<u8> = indices.par_iter().map(|&i| self.read_sequence(i).unwrap()).flatten().collect();
        let sequences: Array2<u8> = Array2::from_shape_vec((indices.len(), self.max_seq_len), sequences).unwrap(); // EPIC BUG!
        let subset = RowMajor::new(sequences, "hamming", true).unwrap();
        Arc::new(subset)
    }

    fn read_sequence(&self, index: Index) -> std::result::Result<Vec<u8>, String> {
        let mut reader = std::io::BufReader::new(File::open(self.path.clone()).unwrap());

        let record = self.records[index].clone();
        let (start, stop) = (0, record.len);

        if stop > record.len {
            return Err("Read interval was out of bounds".to_string());
        } else if start > stop {
            return Err("Invalid query interval".to_string());
        }

        let mut bases_left = stop - start;

        let mut line_offset = start % record.line_bases;
        let line_start = start / record.line_bases * record.line_bytes;
        let offset = record.offset + line_start + line_offset;
        reader.seek(std::io::SeekFrom::Start(offset)).unwrap();

        let mut seq = Vec::new();
        while bases_left > 0 {
            bases_left -= self.read_line(&mut reader, &record, &mut line_offset, bases_left, &mut seq)?;
        }
        if seq.len() < self.max_seq_len {
            seq.extend(vec![0; self.max_seq_len - seq.len()].iter());
        }
        Ok(seq)
    }

    fn read_line(
        &self,
        reader: &mut BufReader<File>,
        record: &FastaRecord,
        line_offset: &mut u64,
        bases_left: u64,
        seq: &mut Vec<u8>,
    ) -> std::result::Result<u64, String> {
        let (bytes_to_read, bytes_to_keep) = {
            let src = reader.fill_buf().expect("Could not get contents of buffer");
            if src.is_empty() {
                return Err("FASTA file is truncated.".to_string());
            }

            let bases_on_line = record.line_bases - min(record.line_bases, *line_offset);
            let bases_in_buffer = min(src.len() as u64, bases_on_line);

            let (bytes_to_read, bytes_to_keep) = if bases_in_buffer <= bases_left {
                let bytes_to_read = min(src.len() as u64, record.line_bytes - *line_offset);

                (bytes_to_read, bases_in_buffer)
            } else {
                (bases_left, bases_left)
            };

            seq.extend_from_slice(&src[..bytes_to_keep as usize]);
            (bytes_to_read, bytes_to_keep)
        };

        reader.consume(bytes_to_read as usize);

        assert!(bytes_to_read > 0);
        *line_offset += bytes_to_read;
        if *line_offset >= record.line_bytes {
            *line_offset = 0;
        }

        Ok(bytes_to_keep)
    }

    fn calculate_distances(&self, left: &[Index], right: &[Index]) -> Array2<u64> {
        let mut distances = Vec::new();
        for &l in left.iter() {
            let mut row = Vec::new();
            let l_instance = self.instance(l);
            for right_chunk in right.chunks(self.batch_size) {
                let right_instances: Vec<Vec<u8>> = right_chunk.par_iter().map(|&r| self.instance(r)).collect();
                row.append(
                    &mut right_chunk
                        .par_iter()
                        .zip(right_instances.par_iter())
                        .map(|(&r, r_instance)| if l == r { 0 } else { self.metric.distance(&l_instance, r_instance) })
                        .collect(),
                );
            }
            distances.push(row);
        }
        let distances: Array1<u64> = distances.into_iter().flatten().collect();
        distances.into_shape((left.len(), right.len())).unwrap()
    }
}

impl Dataset<u8, u64> for FastaDataset {
    fn metric(&self) -> Arc<(dyn Metric<u8, u64>)> {
        Arc::clone(&self.metric)
    }

    fn cardinality(&self) -> usize {
        self.num_sequences
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.num_sequences, self.max_seq_len]
    }

    fn indices(&self) -> Vec<Index> {
        (0..self.num_sequences).collect()
    }

    fn instance(&self, index: Index) -> Vec<u8> {
        self.read_sequence(index).unwrap()
    }

    fn distance(&self, left: Index, right: Index) -> u64 {
        if left == right {
            0
        } else {
            self.metric.distance(&self.instance(left), &self.instance(right))
        }
    }

    fn distances_from(&self, left: Index, right: &[Index]) -> Array1<u64> {
        self.calculate_distances(&[left], right).row(0).to_owned()
    }

    fn distances_among(&self, left: &[Index], right: &[Index]) -> Array2<u64> {
        self.calculate_distances(left, right)
    }

    fn pairwise_distances(&self, indices: &[Index]) -> Array2<u64> {
        self.calculate_distances(indices, indices)
    }
}

impl CompressibleDataset<u8, u64> for FastaDataset {
    fn as_dataset(&self) -> &dyn Dataset<u8, u64> {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use clam::prelude::*;

    use super::FastaDataset;

    #[test]
    fn test_fasta_index() {
        let path = Path::new("/home/nishaq/Documents/research/data/silva-SSU-Ref.fasta");
        let fasta_dataset: Arc<dyn Dataset<u8, u64>> = Arc::new(FastaDataset::new(path).unwrap());

        println!("{}", fasta_dataset.instance(0).len());
    }
}
