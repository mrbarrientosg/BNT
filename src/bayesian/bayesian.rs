use crate::scenario::scenario::Scenario;
use crate::utils::*;
use daggy::petgraph::algo::toposort;

use daggy::petgraph::dot::{Config, Dot};
use daggy::{Dag, NodeIndex, Walker};
use factorial::Factorial;
use itertools::Itertools;
use rand::Rng;
use std::cell::RefCell;

use std::rc::Rc;

use super::population::Population;

#[derive(Debug)]
pub(crate) struct BayesianNode {
    pub(crate) index: usize,
    pub(crate) name: String,
}

pub(crate) struct BayesianNetwork<'a> {
    graph: Dag<BayesianNode, usize, usize>,
    size: usize,
    scenario: &'a Scenario,
    population: Rc<RefCell<Population>>,
}

impl<'a> BayesianNetwork<'a> {
    pub(crate) fn new(scenario: &'a Scenario, population: Rc<RefCell<Population>>) -> Self {
        let mut graph = Dag::<BayesianNode, usize, usize>::new();

        for (i, param) in scenario.parameters().into_iter().enumerate() {
            graph.add_node(BayesianNode {
                index: i,
                name: param.name.clone(),
            });
        }

        Self {
            graph,
            size: scenario.parameters().nb_params(),
            scenario,
            population: population,
        }
    }

    pub(crate) fn dag(&self) -> &Dag<BayesianNode, usize, usize> {
        &self.graph
    }

    pub(crate) fn construct_network(&mut self) {
        let mut graph = Dag::<BayesianNode, usize, usize>::new();

        for (i, param) in self.scenario.parameters().into_iter().enumerate() {
            graph.add_node(BayesianNode {
                index: i,
                name: param.name.clone(),
            });
        }

        self.graph = graph;

        let max_edges = self.scenario.parameters().nb_params();
        // let mut gains: Vec<Vec<f64>> = vec![];

        for node in 0..max_edges {
            let mut p_old = self.cooper_herskovits(node.into());
            let parent_size = self.graph.parents(node.into()).iter(&self.graph).count();

            let mut proceed = true;

            while proceed && parent_size < 2 {
                let mut p_max: Vec<f64> = vec![];
                let pred = &self.scenario.parameters().params[..node];
                let mut candidates_params: Vec<usize> = vec![];

                for (i, p) in pred.iter().enumerate() {
                    let c = self
                        .graph
                        .parents(node.into())
                        .iter(&self.graph)
                        .map(|(_, n)| &self.graph[n])
                        .filter(|&n| n.name == p.name)
                        .count();

                    if c == 0 || self.graph.parents(node.into()).iter(&self.graph).count() == 0 {
                        candidates_params.push(i);
                    }
                }

                if !candidates_params.is_empty() {
                    for j in candidates_params {
                        let index = self
                            .graph
                            .add_edge(NodeIndex::new(j), NodeIndex::new(node), 1)
                            .unwrap();

                        // println!(
                        //     "{:?}",
                        //     Dot::with_config(&self.graph, &[Config::EdgeNoLabel])
                        // );
                        p_max.push(self.cooper_herskovits(node.into()));
                        self.graph.remove_edge(index).unwrap();

                        // println!(
                        //     "{:?}",
                        //     Dot::with_config(&self.graph, &[Config::EdgeNoLabel])
                        // );
                    }
                } else {
                    p_max.push(self.cooper_herskovits(node.into()));
                }

                let from = p_max
                    .iter()
                    .position_max_by(|&a, &b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let p_new = p_max[from];

                if p_new > p_old {
                    p_old = p_new;
                    self.graph
                        .add_edge(NodeIndex::new(from), NodeIndex::new(node), 1)
                        .unwrap();
                } else {
                    proceed = false;
                }

                // println!("{:?}", p_max);
            }

            // let mut max_value = -1.0;
            // let mut from: usize = 0;
            // let mut to: usize = 0;

            // for (i, _) in self.scenario.parameters().into_iter().enumerate() {
            //     let gain = self.compute_gains(NodeIndex::new(i));

            //     gains.push(gain);

            //     for (j, _) in self.scenario.parameters().into_iter().enumerate() {
            //         let current_value = gains[i][j];
            //         if current_value > max_value {
            //             from = i;
            //             to = j;
            //             max_value = current_value;
            //         }
            //     }
            // }

            // if max_value <= 0.0 {
            //     break;
            // }

            // self.graph
            //     .add_edge(NodeIndex::new(from), NodeIndex::new(to), 1).unwrap();

            // gains.clear();
        }
    }

    // fn compute_gains(&self, node_index: NodeIndex<usize>) -> Vec<f64> {
    //     let mut gains: Vec<f64> = vec![-1.0; self.size];
    //     let mut viable: Vec<NodeIndex<usize>> = vec![];

    //     for i in 0..self.graph.node_count() {
    //         let current_node = NodeIndex::new(i);
    //         let mut childrens = self.graph.children(node_index);

    //         if current_node != node_index
    //             && !node_is_children(current_node, &mut childrens, &self.graph)
    //             && !self.path_exists(current_node, node_index)
    //         {
    //             viable.push(current_node);
    //         }
    //     }

    //     for i in 0..self.graph.node_count() {
    //         let candidate_node = NodeIndex::new(i);
    //         let parents_size = self.graph.parents(candidate_node).iter(&self.graph).count();

    //         if parents_size < 2 && viable.contains(&candidate_node) {
    //             gains[i] = self.cooper_herskovits(candidate_node);
    //         } else {
    //             gains[i] = -1.0;
    //         }
    //     }

    //     gains
    // }

    // fn path_exists(&self, i: NodeIndex<usize>, j: NodeIndex<usize>) -> bool {
    //     let mut visited: Vec<NodeIndex<usize>> = vec![];
    //     let mut stack: Vec<NodeIndex<usize>> = vec![];

    //     stack.push(i);

    //     while !stack.is_empty() {
    //         if stack.contains(&j) {
    //             return true;
    //         }

    //         let k = stack.pop().unwrap();

    //         if !visited.contains(&k) {
    //             visited.push(k);

    //             let mut children = self.graph.children(k);

    //             while let Some((_, node)) = children.walk_next(&self.graph) {
    //                 if !visited.contains(&node) {
    //                     stack.push(node);
    //                 }
    //             }
    //         }
    //     }

    //     false
    // }

    fn cooper_herskovits(&self, candidate_node: NodeIndex<usize>) -> f64 {
        let mut prod: f64 = 1.0;

        let parents: Vec<usize> = self
            .graph
            .parents(candidate_node)
            .iter(&self.graph)
            .map(|(_, node)| node.index())
            .collect();

        let uniq: Vec<Vec<usize>> = parents
            .iter()
            .map(|pi| {
                self.population
                    .borrow()
                    .into_iter()
                    .map(|individual| individual.get_index(*pi))
                    .unique()
                    .collect::<Vec<usize>>()
            })
            .collect();

        let phi = find_combinations(&uniq);

        let v_i = self
            .population
            .borrow()
            .into_iter()
            .map(|individual| individual.get_index(candidate_node.index()))
            .unique()
            .collect::<Vec<usize>>();

        let r_i = v_i.len();
        let q_i: usize = phi.len();

        let x_i: Vec<usize> = self
            .population
            .borrow()
            .into_iter()
            .map(|i| i.get_index(candidate_node.index()))
            .collect();

        let x_parents: Vec<Vec<usize>> = parents
            .iter()
            .map(|pi| {
                self.population
                    .borrow()
                    .into_iter()
                    .map(|individual| individual.get_index(*pi))
                    .collect::<Vec<usize>>()
            })
            .collect();

        if q_i > 0 {
            for j in 0..q_i {
                let phi_j = &phi[j];

                let a: Vec<usize> = (0..r_i)
                    .map(|k| self.alpha(k, &v_i, &x_i, &x_parents, Some(phi_j)))
                    .collect();

                let nij: usize = a.iter().sum();

                prod *= (r_i - 1).checked_factorial().unwrap_or(1) as f64
                    / (nij + r_i - 1).checked_factorial().unwrap_or(1) as f64;

                for value in a {
                    prod *= value.checked_factorial().unwrap_or(1) as f64;
                }
            }
        } else {
            let a: Vec<usize> = (0..r_i)
                .map(|k| self.alpha(k, &v_i, &x_i, &x_parents, None))
                .collect();
            let nij: usize = a.iter().sum();

            prod *= (r_i - 1).checked_factorial().unwrap_or(1) as f64
                / (nij + r_i - 1).checked_factorial().unwrap_or(1) as f64;

            for value in a {
                prod *= value.checked_factorial().unwrap_or(1) as f64;
            }
        }

        prod
    }

    fn alpha(
        &self,
        k: usize,
        vi: &Vec<usize>,
        x_i: &Vec<usize>,
        x_parents: &Vec<Vec<usize>>,
        phi: Option<&Vec<usize>>,
    ) -> usize {
        if let Some(phi) = phi {
            let phi_in_parents = self.parents_contains_phi(x_parents, phi);

            x_i.iter()
                .enumerate()
                .filter(|(_, &data)| data == vi[k])
                .map(|(idx, _)| idx)
                .filter(|&idx| phi_in_parents[idx])
                .count()
        } else {
            x_i.iter().filter(|&&data| data == vi[k]).count()
        }
    }

    fn parents_contains_phi(&self, x_parents: &Vec<Vec<usize>>, phi: &Vec<usize>) -> Vec<bool> {
        let mut phi_in_parents: Vec<bool> = vec![false; x_parents[0].len()];

        for i in 0..x_parents[0].len() {
            let mut flag = true;
            for j in 0..x_parents.len() {
                if x_parents[j][i] != phi[j] {
                    flag = false;
                    break;
                }
            }

            phi_in_parents[i] = flag;
        }
        phi_in_parents
    }

    pub(crate) fn sample(&self, nb_samples: usize) -> Vec<Vec<usize>> {
        let mut samples: Vec<Vec<usize>> = vec![];

        let ordered = toposort(self.graph.graph(), None).unwrap();

        for _ in 0..nb_samples {
            let sample = self.probabilistic_sample(&ordered);
            samples.push(sample.clone());
        }

        samples
    }

    fn probabilistic_sample(&self, ordered: &Vec<NodeIndex<usize>>) -> Vec<usize> {
        let mut sample: Vec<usize> = vec![0; self.size];
        let mut rng = rand::thread_rng();
        let mut node_with_parents: Vec<NodeIndex<usize>> = vec![];

        for i in ordered {
            if self.graph.parents(*i).iter(&self.graph).count() == 0 {
                let param_domain_size: usize = self
                    .scenario
                    .parameters()
                    .get_parameter(i.index())
                    .domain_size();

                let prob = self.marginal_probability(*i);

                let mut sumprob: f64 = 0.0;
                let mut index = 0;
                let value: f64 = rng.gen();

                for j in 0..param_domain_size {
                    index = j;
                    sumprob += prob[j];

                    if value <= sumprob {
                        break;
                    }
                }

                sample[i.index()] = index;
            } else {
                node_with_parents.push(*i);
            }
        }

        for i in node_with_parents {
            let param_domain_size: usize = self
                .scenario
                .parameters()
                .get_parameter(i.index())
                .domain_size();

            let prob = self.calculate_probability(i, &sample);

            let mut sumprob: f64 = 0.0;
            let mut index = 0;
            let value: f64 = rng.gen();

            for j in 0..param_domain_size {
                index = j;
                sumprob += prob[j];

                if value <= sumprob {
                    break;
                }
            }

            sample[i.index()] = index;
        }

        sample
    }

    fn calculate_probability(
        &self,
        current_node: NodeIndex<usize>,
        sample: &Vec<usize>,
    ) -> Vec<f64> {
        let parents: Vec<usize> = self
            .graph
            .parents(current_node)
            .iter(&self.graph)
            .map(|(_, node)| node.index())
            .collect();

        let v_i = self
            .population
            .borrow()
            .into_iter()
            .map(|individual| individual.get_index(current_node.index()))
            .unique()
            .collect::<Vec<usize>>();

        let r_i = v_i.len();

        let x_i: Vec<usize> = self
            .population
            .borrow()
            .into_iter()
            .map(|i| i.get_index(current_node.index()))
            .collect();

        let x_parents: Vec<Vec<usize>> = parents
            .iter()
            .map(|pi| {
                self.population
                    .borrow()
                    .into_iter()
                    .map(|individual| individual.get_index(*pi))
                    .collect::<Vec<usize>>()
            })
            .collect();

        let mut prob: Vec<f64> = vec![0.0; r_i];

        let phi_j = parents.iter().map(|&p| sample[p]).collect_vec();

        let num_of_samples_parents = self
            .parents_contains_phi(&x_parents, &phi_j)
            .iter()
            .filter(|&&f| f)
            .count() as f64;

        for k in 0..r_i {
            let num_of_samples_all = self.alpha(k, &v_i, &x_i, &x_parents, Some(&phi_j)) as f64;
            prob[k] = (num_of_samples_all + 1.0) / (num_of_samples_parents + 1.0 * r_i as f64);
        }

        prob
    }

    fn marginal_probability(&self, current_node: NodeIndex<usize>) -> Vec<f64> {
        let param_domain_size: usize = self
            .scenario
            .parameters()
            .get_parameter(current_node.index())
            .domain_size();

        let mut prob: Vec<f64> = vec![0.0; param_domain_size];

        self.population
            .borrow()
            .into_iter()
            .for_each(|i| prob[i.get_index(current_node.index())] += 1.0);

        prob.iter()
            .map(|v| *v / self.population.borrow().population_size() as f64)
            .collect()
    }
}

#[cfg(test)]
mod bayesian_tests {
    use daggy::petgraph::dot::{Config, Dot};

    use crate::{
        bayesian::{
            bayesian::BayesianNetwork,
            population::{Individual, Population},
        },
        scenario::{
            parameter::{Domain, Parameter},
            scenario::Scenario,
        },
    };

    #[test]
    fn network_creation() {
        let scenario = Scenario::new()
            .add_parameter(Parameter::new("x1", "x1", Domain::Integer(0, 1)))
            .add_parameter(Parameter::new("x2", "x2", Domain::Integer(0, 1)))
            .add_parameter(Parameter::new("x3", "x3", Domain::Integer(0, 1)));

        let population = Population::new(10, 10);
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![1, 0, 0]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![1, 1, 1]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![0, 0, 1]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![1, 1, 1]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![0, 0, 0]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![0, 1, 1]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![1, 1, 1]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![0, 0, 0]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![1, 1, 1]));
        population
            .borrow_mut()
            .individuals
            .push(Individual::from_sample(&vec![0, 0, 0]));

        let mut bayesian = BayesianNetwork::new(&scenario, population);
        bayesian.construct_network();

        println!(
            "{:?}",
            Dot::with_config(&bayesian.graph, &[Config::EdgeNoLabel])
        );
    }
}
