use std::path::Path;
use std::sync::Arc;

use clam::Cakes;

use clam_bio::FastaDataset;

fn main() {
    let fasta_path = Path::new("/home/nishaq/Documents/research/data/silva-SSU-Ref.fasta");

    let reading_time = std::time::Instant::now();
    let fasta_dataset = Arc::new(FastaDataset::new(fasta_path).unwrap());
    println!("{:.2e} seconds to read dataset.", reading_time.elapsed().as_secs_f64());

    let subset_time = std::time::Instant::now();
    let subset_indices = fasta_dataset.subsample_indices(25_000);
    let row_major_subset = fasta_dataset.get_subset_from_indices(&subset_indices);
    println!("{:.2e} seconds to create subset.", subset_time.elapsed().as_secs_f64());

    let cakes_time = std::time::Instant::now();
    let cakes = Cakes::build(row_major_subset, Some(20), Some(100));
    println!("{:.2e} seconds to create cakes index.", cakes_time.elapsed().as_secs_f64());
    println!("tree diameter {:.2e}", cakes.diameter());
}
