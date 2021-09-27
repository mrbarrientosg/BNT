use daggy::{Children, Dag, NodeIndex, Walker};

use crate::bayesian::bayesian::BayesianNode;

pub(crate) fn node_is_children(
    node: NodeIndex<usize>,
    children: &mut Children<BayesianNode, usize, usize>,
    dag: &Dag<BayesianNode, usize, usize>,
) -> bool {
    while let Some((_, child)) = children.walk_next(&dag) {
        if node == child {
            return true;
        }
    }
    false
}

pub(crate) fn find_combinations(arr: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    if arr.is_empty() {
        return vec![];
    }

    let mut indices = arr.iter().map(|ref _x| 0).collect();
    let mut paths: Vec<Vec<usize>> = vec![];

    _find_combinations(&arr, &mut indices, 0, &mut paths).to_vec()
}

fn _find_combinations<'a>(
    arr: &Vec<Vec<usize>>,
    indices: &mut Vec<usize>,
    index: usize,
    paths: &'a mut Vec<Vec<usize>>,
) -> &'a Vec<Vec<usize>> {
    for (i, _) in arr[index].iter().enumerate() {
        let i = i;
        indices[index] = i;
        if index == arr.len() - 1 {
            paths.push(indices.clone());
        }

        if index < arr.len() - 1 {
            _find_combinations(arr, indices, index + 1, paths);
        }
    }
    paths
}
