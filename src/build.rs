use std::collections::HashMap;
use std::sync::Arc;

use bitvec::prelude::*;
use clam::prelude::*;
use rayon::prelude::*;

type TreeMap<T, U> = HashMap<BitVec<Lsb0, u8>, Arc<Cluster<T, U>>>;

/// Given a Cluster, unstack the tree into a HashSet of Clusters
/// and erase all parent-child relationships.
pub fn unstack_tree<T: Number, U: Number>(root: &Arc<Cluster<T, U>>) -> TreeMap<T, U> {
    let mut tree = root.flatten_tree();
    tree.push(Arc::clone(root));

    tree.par_iter()
        .map(|c| {
            let key = c.name.clone();
            let cluster = Arc::new(Cluster {
                dataset: Arc::clone(&c.dataset),
                name: c.name.clone(),
                cardinality: c.cardinality,
                indices: c.indices.clone(),
                children: None,
                argsamples: c.argsamples.clone(),
                argcenter: c.argcenter,
                argradius: c.argradius,
                radius: c.radius,
            });
            (key, cluster)
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
pub fn restack_tree<T: Number, U: Number>(tree: TreeMap<T, U>) -> Arc<Cluster<T, U>> {
    let depth = tree.par_iter().map(|(_, c)| c.depth()).max().unwrap();
    let mut tree = tree;

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


/// For every cluster in the tree, replace the clusters' dataset with the given dataset.
pub fn replace_dataset_in_tree<T: Number, U: Number>(root: Arc<Cluster<T, U>>, dataset: Arc<dyn Dataset<T, U>>) -> Arc<Cluster<T, U>> {
    let children = match &root.children {
        Some((left, right)) => {
            let left = replace_dataset_in_tree(Arc::clone(left), Arc::clone(&dataset));
            let right = replace_dataset_in_tree(Arc::clone(right), Arc::clone(&dataset));
            Some((left, right))
        },
        None => None,
    };

    Arc::new(Cluster {
        dataset,
        name: root.name.clone(),
        cardinality: root.cardinality,
        indices: root.indices.clone(),
        argsamples: root.argsamples.clone(),
        argcenter: root.argcenter,
        argradius: root.argradius,
        radius: root.radius,
        children,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use clam::prelude::*;
    use clam::Cakes;

    use crate::build::{restack_tree, unstack_tree};
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
        assert_eq!(0, unstacked.iter().filter(|(_, c)| c.children.is_some()).count());

        let restacked = &restack_tree(unstacked);
        assert_eq!(root.num_descendants(), restacked.num_descendants());
        assert_eq!(root, restacked);
        assert_eq!(root.flatten_tree(), restacked.flatten_tree());

        println!("{}", fasta_dataset.instance(0).len());
    }
}
