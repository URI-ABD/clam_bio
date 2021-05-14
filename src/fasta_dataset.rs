use std::cmp::min;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufReader, Seek, BufRead};
use std::path::Path;
use std::sync::{Arc, Mutex};

use clam::prelude::*;
use clam::CompressibleDataset;
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
    pub num_sequences: usize,
    pub max_seq_len: usize,
    metric: Arc<dyn Metric<u8, u64>>,
    cache: Mutex<HashMap<(usize, usize), u64>>,
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

        Ok(FastaDataset {
            path: fasta_path.as_os_str().to_owned(),
            records,
            num_sequences,
            max_seq_len,
            metric: Arc::new(clam::metric::Hamming),
            cache: Mutex::new(HashMap::new()),
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
            let key = if left < right { (left, right) } else { (right, left) };
            if !self.cache.lock().unwrap().contains_key(&key) {
                let distance = self.metric.distance(&self.instance(left), &self.instance(right));
                self.cache.lock().unwrap().insert(key, distance);
                distance
            } else {
                *self.cache.lock().unwrap().get(&key).unwrap()
            }
        }
    }
}

impl CompressibleDataset<u8, u64> for FastaDataset {
    fn as_dataset(&self) -> &dyn Dataset<u8, u64> {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::path::Path;

    use clam::prelude::*;

    use super::FastaDataset;

    #[test]
    fn test_fasta_index() {
        let path = Path::new("/data/abd/silva/silva-SSU-Ref.fasta");
        let fasta_dataset: Arc<dyn Dataset<u8, u64>> = Arc::new(FastaDataset::new(path).unwrap());

        println!("{}", fasta_dataset.instance(0).len());
    }
}
