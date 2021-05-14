use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use bio::io::fasta::IndexedReader;
use clam::prelude::*;

struct FastaDataset {
    reader: Mutex<IndexedReader<std::fs::File>>,
    pub num_sequences: usize,
    pub seq_len: usize,
    metric: Arc<dyn Metric<u8, u64>>,
    cache: Mutex<HashMap<(usize, usize), u64>>,
}

impl std::fmt::Debug for FastaDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("RowMajor Dataset")
            .field("num_sequences", &self.num_sequences)
            .field("seq_len", &self.seq_len)
            .field("metric", &self.metric.name())
            .finish()
    }
}

impl FastaDataset {
    pub fn new(fasta_path: &Path) -> std::result::Result<FastaDataset, String> {
        let mut reader = IndexedReader::from_file(&fasta_path).expect("could not read from given file.");
        let num_sequences = reader.index.sequences().len();
        let mut seq = vec![];
        reader.fetch_all_by_rid(0).expect("could not fetch the first sequence");
        reader.read(&mut seq).expect("could not read the first sequence into a vec");
        let seq_len = seq.len();

        Ok(FastaDataset {
            reader: Mutex::new(reader),
            num_sequences,
            seq_len,
            metric: Arc::new(clam::metric::Hamming),
            cache: Mutex::new(HashMap::new()),
        })
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
        vec![self.num_sequences, self.seq_len]
    }

    fn indices(&self) -> Indices {
        (0..self.num_sequences).collect()
    }

    fn instance(&self, index: Index) -> Vec<u8> {
        let mut reader = self.reader.lock().unwrap();
        let mut seq = Vec::new();
        reader.fetch_all_by_rid(index).unwrap();
        reader.read(&mut seq).unwrap();
        seq
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

fn main() {
    let fasta_path = Path::new("/data/abd/silva/silva-SSU-Ref.fasta");
    let reader = FastaDataset::new(fasta_path).unwrap();
    println!("{:?}", reader.instance(0));
}
