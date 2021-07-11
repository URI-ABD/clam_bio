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

                let (children, indices) = if leaves.contains_key(&left_name) {
                    let left = Arc::clone(leaves.get(&left_name).unwrap());
                    let right = Arc::clone(leaves.get(&right_name).unwrap());

                    let mut indices = left.indices.clone();
                    indices.append(&mut right.indices.clone());

                    (Some((left, right)), indices)
                } else {
                    (None, cluster.indices.clone())
                };

                let cluster = Arc::new(Cluster {
                    dataset: Arc::clone(&cluster.dataset),
                    name: cluster.name.clone(),
                    cardinality: indices.len(),
                    indices,
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
fn add_instance<T: Number, U: Number>(cluster: &Arc<Cluster<T, U>>, sequence: &[T], distance: U) -> RadiusMap<T, U> {
    match &cluster.children {
        Some((left, right)) => {
            let left_distance = cluster.dataset.metric().distance(&left.center(), sequence);
            let right_distance = cluster.dataset.metric().distance(&right.center(), sequence);

            if left_distance <= right_distance {
                add_instance(left, sequence, left_distance)
            } else {
                add_instance(right, sequence, right_distance)
            }
        }
        None => [(Arc::clone(cluster), distance)].iter().cloned().collect(),
    }
}

pub fn build_cakes_from_fasta(
    fasta_dataset: &Arc<FastaDataset>,
    subsample_size: usize,
    max_depth: Option<usize>,
    min_cardinality: Option<usize>,
) -> Cakes<u8, u64> {
    let subset_indices = fasta_dataset.subsample_indices(subsample_size);

    // TODO: Add from complement in batches
    let complement_indices = fasta_dataset.get_complement_indices(&subset_indices)[..subsample_size].to_vec();

    let row_major_subset = fasta_dataset.get_subset_from_indices(&subset_indices);

    let cakes = Cakes::build(row_major_subset, max_depth, min_cardinality);

    let flat_tree = {
        let mut tree = cakes.root.flatten_tree();
        tree.push(Arc::clone(&cakes.root));
        tree
    };

    // Build a sparse matrix of cluster insertions.
    // | Sequence | Cluster 0                   | Cluster 1  |
    // | seq_00   | None (Not added to cluster) | Some(dist) |
    // | seq_01   | Some(dist)                  | None       |
    let insertion_paths: Vec<Vec<Option<u64>>> = complement_indices
        .par_iter()
        .map(|&index| {
            let sequence = fasta_dataset.instance(index);
            let distance = cakes.root.dataset.metric().distance(&cakes.root.center(), &sequence);
            let insertion_path = add_instance(&cakes.root, &sequence, distance);

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

    // Reduce the matrix to find the maximum
    let new_radii: Vec<_> = flat_tree
        .par_iter()
        .enumerate()
        .map(|(i, _)| {
            let temp: Vec<_> = insertion_paths.par_iter().map(|inner| inner[i].unwrap_or(0)).collect();
            clam::utils::argmax(&temp)
        })
        .collect();

    let insertions: Vec<Vec<usize>> = flat_tree
        .par_iter()
        .enumerate()
        .map(|(i, _)| {
            let temp: Vec<Option<usize>> = insertion_paths
                .par_iter()
                .enumerate()
                .map(|(j, inner)| {
                    let distance = inner[i];
                    if distance.is_some() {
                        Some(j)
                    } else {
                        None
                    }
                })
                .collect();
            temp.into_par_iter().filter(|&v| v.is_some()).map(|v| v.unwrap()).collect()
        })
        .collect();

    let dataset = Arc::clone(fasta_dataset).as_arc_dataset();

    let unstacked_tree = flat_tree
        .into_par_iter()
        .zip(new_radii.into_par_iter())
        .zip(insertions.into_par_iter())
        .map(|((cluster, (argradius, radius)), indices)| {
            let indices = {
                let mut indices: Vec<_> = indices.into_iter().map(|i| complement_indices[i]).collect();
                indices.extend(cluster.indices.iter().map(|&i| subset_indices[i]));
                indices
            };

            let argsamples: Vec<_> = cluster.argsamples.iter().map(|&i| subset_indices[i]).collect();
            let argcenter = subset_indices[cluster.argcenter];

            let (argradius, radius) = if radius > cluster.radius {
                (complement_indices[argradius], radius)
            } else {
                (subset_indices[cluster.argradius], cluster.radius)
            };

            Arc::new(Cluster {
                dataset: Arc::clone(&dataset),
                name: cluster.name.clone(),
                cardinality: indices.len(),
                indices,
                argsamples,
                argcenter,
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

    use crate::build::*;
    use crate::FastaDataset;

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
