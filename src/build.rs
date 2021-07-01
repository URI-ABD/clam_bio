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

fn left_child_name<T: Number, U: Number>(cluster: &Arc<Cluster<T, U>>) -> BitVec<Lsb0, u8> {
    let mut name = cluster.name.clone();
    name.push(false);
    name
}

fn right_child_name<T: Number, U: Number>(cluster: &Arc<Cluster<T, U>>) -> BitVec<Lsb0, u8> {
    let mut name = cluster.name.clone();
    name.push(true);
    name
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
            .iter()
            .map(|(_, c)| {
                let left_name = left_child_name(c);
                let right_name = right_child_name(c);

                let children = if leaves.contains_key(&left_name) {
                    let left = Arc::clone(leaves.get(&left_name).unwrap());
                    let right = Arc::clone(leaves.get(&right_name).unwrap());
                    Some((left, right))
                } else {
                    None
                };

                let cluster = Arc::new(Cluster {
                    dataset: Arc::clone(&c.dataset),
                    name: c.name.clone(),
                    cardinality: c.cardinality,
                    indices: c.indices.clone(),
                    children,
                    argsamples: c.argsamples.clone(),
                    argcenter: c.argcenter,
                    argradius: c.argradius,
                    radius: c.radius,
                });

                (cluster.name.clone(), cluster)
            })
            .collect();

        parents.into_iter().for_each(|(k, v)| {
            ancestors.insert(k, v);
        });

        tree = ancestors;
    }

    assert_eq!(1, tree.len());
    Arc::clone(tree.get(&bitvec![Lsb0, u8; 1]).unwrap())
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

        let restacked = &restack_tree(unstacked);
        let restacked_str = vec![
            "Cluster {".to_string(),
            format!("cardinality: {:?},", restacked.cardinality),
            format!("num_indices: {:?},", restacked.indices.len()),
            format!("argcenter: {:?},", restacked.argcenter),
            format!("argradius: {:?},", restacked.argradius),
            format!("radius: {:?},", restacked.radius),
            format!("has children: {:?}", restacked.children.is_some()),
            "}".to_string(),
        ]
        .join(" ");

        let root_str = vec![
            "Cluster {".to_string(),
            format!("cardinality: {:?},", root.cardinality),
            format!("num_indices: {:?},", root.indices.len()),
            format!("argcenter: {:?},", root.argcenter),
            format!("argradius: {:?},", root.argradius),
            format!("radius: {:?},", root.radius),
            format!("has children: {:?}", root.children.is_some()),
            "}".to_string(),
        ]
        .join(" ");

        assert_eq!(root_str, restacked_str);

        println!("{}", fasta_dataset.instance(0).len());
    }
}
