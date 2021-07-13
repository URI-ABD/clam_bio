use std::path::Path;
use std::sync::Arc;

use clam::{Cakes, Dataset};

use clam_bio::FastaDataset;

fn main() {
    let path = Path::new("/data/abd/silva/silva-SSU-Ref.fasta");
    let dataset: Arc<dyn Dataset<u8, u64>> = Arc::new(FastaDataset::new(path).unwrap());

    let start_time = std::time::Instant::now();
    // let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    // let cakes = pool.install(|| Cakes::build_in_batches(
    //     Arc::clone(&dataset), 
    //     Some(0.1), 
    //     Some(50), 
    //     Some(100),
    // ));
    let cakes = Cakes::build_in_batches(
        Arc::clone(&dataset), 
        Some(0.8), 
        Some(50), 
        Some(100),
    );
    println!("{:.2e} seconds to create cakes from subset.", start_time.elapsed().as_secs_f64());

    assert_eq!(cakes.root.cardinality, dataset.cardinality());

    println!("cakes diameter: {}", cakes.diameter());
}
