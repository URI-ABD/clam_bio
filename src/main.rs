use std::path::Path;
use std::sync::Arc;

use clam::prelude::*;
use clam::Cakes;

use clam_bio::FastaDataset;

fn main() {
    let fasta_path = Path::new("/data/abd/silva/silva-SSU-Ref.fasta");

    let reading_time = std::time::Instant::now();
    let fasta_dataset: Arc<dyn Dataset<u8, u64>> = Arc::new(FastaDataset::new(fasta_path).unwrap());
    println!("{:.2e} seconds to read dataset.", reading_time.elapsed().as_secs_f64());

    let cakes_time = std::time::Instant::now();
    let cakes = Cakes::build(fasta_dataset, Some(20), Some(100));
    println!("{:.2e} seconds to create cakes index.", cakes_time.elapsed().as_secs_f64());
    println!("{:.2e}", cakes.diameter());
}
