use std::cmp::{min, max};
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek};
use std::path::Path;
use std::sync::{Arc, Mutex};

use clam::prelude::*;
use clam::CompressibleDataset;
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
    cache: Mutex<HashMap<(usize, usize), u64>>,
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
            cache: Mutex::new(HashMap::new()),
            batch_size,
        })
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

    fn calculate_distances(&self, left: &[Index], right: &[Index]) {
        left.chunks(self.batch_size).for_each(|l_chunk| {
            let l_instances: Vec<Vec<u8>> = l_chunk.iter().map(|&l| self.instance(l)).collect();
            right.chunks(self.batch_size).for_each(|r_chunk| {
                let r_instances: Vec<Vec<u8>> = r_chunk.iter().map(|&r| self.instance(r)).collect();
                l_chunk.par_iter().zip(l_instances.par_iter()).for_each(|(&l, li)| {
                    r_chunk.par_iter().zip(r_instances.par_iter()).filter(|(&r, _)| l != r).for_each(|(&r, ri)| {
                        let key = if l < r { (l, r) } else { (r, l) };
                        self.cache.lock().unwrap().entry(key).or_insert_with(|| self.metric.distance(li, ri));
                    })
                })
            })
        });
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

    fn distances_from(&self, left: Index, right: &[Index]) -> Vec<u64> {
        self.calculate_distances(&[left], right);
        right
            .par_iter()
            .map(|&r| {
                if left == r {
                    0
                } else {
                    let key = if left < r { (left, r) } else { (r, left) };
                    *self.cache.lock().unwrap().get(&key).unwrap()
                }
            })
            .collect()
    }

    fn distances_among(&self, left: &[Index], right: &[Index]) -> Vec<Vec<u64>> {
        self.calculate_distances(left, right);
        left.par_iter().map(|&l| self.distances_from(l, right)).collect()
    }

    fn pairwise_distances(&self, indices: &[Index]) -> Vec<Vec<u64>> {
        self.distances_among(indices, indices)
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
        let path = Path::new("/data/abd/silva/silva-SSU-Ref.fasta");
        let fasta_dataset: Arc<dyn Dataset<u8, u64>> = Arc::new(FastaDataset::new(path).unwrap());

        println!("{}", fasta_dataset.instance(0).len());
    }
}
