use std::path::Path;
use std::sync::Arc;

use clam_bio::build::build_cakes_from_fasta;
use clam_bio::FastaDataset;

fn main() {
    let path = Path::new("/home/nishaq/Documents/research/data/silva-SSU-Ref.fasta");
    let fasta_dataset = Arc::new(FastaDataset::new(path).unwrap());
    let subsample_size = 25_000;

    let start_time = std::time::Instant::now();
    let cakes = build_cakes_from_fasta(&fasta_dataset, subsample_size, Some(50), None);
    println!("{:.2e} seconds to create cakes from subset.", start_time.elapsed().as_secs_f64());

    assert_eq!(cakes.root.cardinality, 2 * subsample_size);

    println!("cakes diameter: {}", cakes.diameter());
}
