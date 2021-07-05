use std::collections::HashMap;
use std::sync::Arc;

use bitvec::prelude::*;
use clam::prelude::*;
use clam::Cakes;
use rayon::prelude::*;

use crate::FastaDataset;

type TreeMap<T, U> = HashMap<BitVec<Lsb0, u8>, Arc<Cluster<T, U>>>;
type TreeVec<T, U> = Vec<Arc<Cluster<T, U>>>;
type RadiusMap<T, U> = HashMap<Arc<Cluster<T, U>>, U>;

/// Given a Cluster, unstack the tree into a HashSet of Clusters
/// and erase all parent-child relationships.
pub fn unstack_tree<T: Number, U: Number>(root: &Arc<Cluster<T, U>>) -> TreeVec<T, U> {
    let mut tree = root.flatten_tree();
    tree.push(Arc::clone(root));

    tree.par_iter()
        .map(|c| {
            Arc::new(Cluster {
                dataset: Arc::clone(&c.dataset),
                name: c.name.clone(),
                cardinality: c.cardinality,
                indices: c.indices.clone(),
                children: None,
                argsamples: c.argsamples.clone(),
                argcenter: c.argcenter,
                argradius: c.argradius,
                radius: c.radius,
            })
        })
        .collect()
}

fn child_names<T: Number, U: Number>(cluster: &Arc<Cluster<T, U>>) -> (BitVec<Lsb0, u8>, BitVec<Lsb0, u8>) {
    let mut left_name = cluster.name.clone();
    left_name.push(false);

    let mut right_name = cluster.name.clone();
    right_name.push(true);

    (left_name, right_name)
}

/// Given an unstacked tree as a HashMap of Clusters, rebuild all
/// parent-child relationships and return the root cluster.
/// This consumed the given HashMap.
pub fn restack_tree<T: Number, U: Number>(tree: TreeVec<T, U>) -> Arc<Cluster<T, U>> {
    let depth = tree.par_iter().map(|c| c.depth()).max().unwrap();
    let mut tree: TreeMap<T, U> = tree.into_par_iter().map(|c| (c.name.clone(), c)).collect();

    for d in (0..depth).rev() {
        let (leaves, mut ancestors): (TreeMap<T, U>, TreeMap<T, U>) = tree.drain().partition(|(_, v)| v.depth() == d + 1);
        let (parents, mut ancestors): (TreeMap<T, U>, TreeMap<T, U>) = ancestors.drain().partition(|(_, v)| v.depth() == d);

        let parents: TreeMap<T, U> = parents
            .par_iter()
            .map(|(_, cluster)| {
                let (left_name, right_name) = child_names(cluster);

                let children = if leaves.contains_key(&left_name) {
                    let left = Arc::clone(leaves.get(&left_name).unwrap());
                    let right = Arc::clone(leaves.get(&right_name).unwrap());
                    Some((left, right))
                } else {
                    None
                };

                let cluster = Arc::new(Cluster {
                    dataset: Arc::clone(&cluster.dataset),
                    name: cluster.name.clone(),
                    cardinality: cluster.cardinality,
                    indices: cluster.indices.clone(),
                    children,
                    argsamples: cluster.argsamples.clone(),
                    argcenter: cluster.argcenter,
                    argradius: cluster.argradius,
                    radius: cluster.radius,
                });

                (cluster.name.clone(), cluster)
            })
            .collect();

        parents.into_iter().for_each(|(name, cluster)| {
            ancestors.insert(name, cluster);
        });

        tree = ancestors;
    }

    assert_eq!(1, tree.len());
    Arc::clone(tree.get(&bitvec![Lsb0, u8; 1]).unwrap())
}

// TODO: Measure if this is faster done in batches
fn add_instance<T: Number, U: Number>(cluster: &Arc<Cluster<T, U>>, sequence: &[T]) -> RadiusMap<T, U> {
    match &cluster.children {
        Some((left, right)) => {
            let left_distance = cluster.dataset.metric().distance(&left.center(), sequence);
            let right_distance = cluster.dataset.metric().distance(&right.center(), sequence);

            if left_distance <= right_distance {
                let mut result = add_instance(left, sequence);
                result.insert(Arc::clone(left), left_distance);
                result
            } else {
                let mut result = add_instance(right, sequence);
                result.insert(Arc::clone(right), right_distance);
                result
            }
        }
        None => HashMap::new(),
    }
}

pub fn build_cakes_from_fasta(
    fasta_dataset: &Arc<FastaDataset>,
    subsample_size: usize,
    max_depth: Option<usize>,
    min_cardinality: Option<usize>,
) -> Cakes<u8, u64> {
    let subset_indices = fasta_dataset.subsample_indices(subsample_size);
    let complement_indices = fasta_dataset.get_complement_indices(&subset_indices);

    let row_major_subset = fasta_dataset.get_subset_from_indices(&subset_indices);
    let cakes = Cakes::build(row_major_subset, max_depth, min_cardinality);

    let flat_tree = {
        let mut tree = cakes.root.flatten_tree();
        tree.push(Arc::clone(&cakes.root));
        tree
    };
    // Builds a matrix of values
    //   
    let insertion_paths: Vec<Vec<Option<u64>>> = complement_indices
        .par_iter()
        .map(|&index| {
            let sequence = fasta_dataset.instance(index);

            let mut insertion_path = add_instance(&cakes.root, &sequence);
            let distance = cakes.root.dataset.metric().distance(&cakes.root.center(), &sequence);
            insertion_path.insert(Arc::clone(&cakes.root), distance);

            flat_tree
                .par_iter()
                .map(|cluster| {
                    if insertion_path.contains_key(cluster) {
                        Some(*insertion_path.get(cluster).unwrap())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    let new_radii: Vec<(Index, u64)> = flat_tree
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let temp: Vec<u64> = insertion_paths
                .iter()
                .map(|inner| {
                    let distance = inner[i];
                    if distance.is_some() {
                        distance.unwrap()
                    } else {
                        0
                    }
                })
                .collect();
            clam::utils::argmax(&temp)
        })
        .collect();

    let insertions: Vec<Vec<Index>> = flat_tree
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let temp: Vec<Option<Index>> = insertion_paths
                .iter()
                .map(|inner| {
                    let distance = inner[i];
                    if distance.is_some() {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            temp.iter().filter(|&&v| v.is_some()).map(|v| v.unwrap()).collect()
        })
        .collect();

    let dataset = Arc::clone(fasta_dataset).as_arc_dataset();

    let unstacked_tree = flat_tree
        .into_iter()
        .zip(new_radii.into_iter())
        .zip(insertions.into_iter())
        .map(|((cluster, (new_argradius, new_radius)), mut inserted_indices)| {
            let mut indices = cluster.indices.clone();
            indices.append(&mut inserted_indices);

            let (argradius, radius) = if new_radius > cluster.radius {
                (new_argradius, new_radius)
            } else {
                (cluster.argradius, cluster.radius)
            };

            Arc::new(Cluster {
                dataset: Arc::clone(&dataset),
                name: cluster.name.clone(),
                cardinality: indices.len(),
                indices,
                argsamples: cluster.argsamples.clone(),
                argcenter: cluster.argcenter,
                argradius,
                radius,
                children: None,
            })
        })
        .collect();

    Cakes {
        dataset: cakes.dataset,
        root: restack_tree(unstacked_tree),
        metric: cakes.metric,
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use clam::prelude::*;
    use clam::Cakes;

    use crate::build::*;
    use crate::FastaDataset;

    #[test]
    fn test_stack_unstack() {
        let path = Path::new("/home/nishaq/Documents/research/data/silva-SSU-Ref.fasta");
        let fasta_dataset = Arc::new(FastaDataset::new(path).unwrap());

        let subset_indices = fasta_dataset.subsample_indices(1024);
        let row_major_subset = fasta_dataset.get_subset_from_indices(&subset_indices);

        let cakes = Cakes::build(row_major_subset, Some(8), None);
        let root = &cakes.root;

        let unstacked = unstack_tree(root);
        assert_eq!(unstacked.len(), 1 + root.num_descendants());
        assert_eq!(0, unstacked.iter().filter(|c| c.children.is_some()).count());

        let restacked = &restack_tree(unstacked);
        assert_eq!(root.num_descendants(), restacked.num_descendants());
        assert_eq!(root, restacked);
        assert_eq!(root.flatten_tree(), restacked.flatten_tree());

        println!("{}", fasta_dataset.instance(0).len());
    }

    #[test]
    fn test_cakes_from_subsample() {
        let path = Path::new("/home/nishaq/Documents/research/data/silva-SSU-Ref.fasta");
        let fasta_dataset = Arc::new(FastaDataset::new(path).unwrap());
        let subsample_size = 1024;

        let start_time = std::time::Instant::now();
        let cakes = build_cakes_from_fasta(&fasta_dataset, subsample_size, Some(8), None);
        println!("{:.2e} seconds to create cakes from subset.", start_time.elapsed().as_secs_f64());

        assert_eq!(cakes.root.cardinality, 2 * subsample_size);
    }
}
