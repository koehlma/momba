use std::sync::Arc;

use hashbrown::HashMap;
//use std::collections::HashMap;
use momba_explore::*;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use tch::{
    kind::*,
    nn,
    nn::{Linear, Module, Sequential},
    Tensor,
};

use crate::simulate::Oracle;

use self::generic::{ActionResolver, EdgeByIndexResolver};

mod generic;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Layers>,
}

impl NeuralNetwork {
    pub fn get_input_size(&self) -> i64 {
        match self.layers.first().unwrap() {
            Layers::Linear {
                name: _,
                input_size,
                output_size: _,
                has_biases: _,
                weights: _,
                biases: _,
            } => *input_size,
            _ => 0,
        }
    }
    pub fn get_output_size(&self) -> i64 {
        match self.layers.last().unwrap() {
            Layers::Linear {
                name: _,
                input_size: _,
                output_size,
                has_biases: _,
                weights: _,
                biases: _,
            } => *output_size,
            _ => 0,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "kind")]
enum Layers {
    // There can be much more different types of NN
    #[serde(rename_all = "camelCase")]
    Linear {
        name: String,
        input_size: i64,
        output_size: i64,
        has_biases: bool,
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
    },
    ReLU {
        name: String,
    },
}

struct ModelWrapper {
    pub _model: Sequential,
    pub _nn: Arc<NeuralNetwork>,
}
impl ModelWrapper {
    fn new(nn: Arc<NeuralNetwork>) -> Self {
        let mut tch_nn = nn::seq();
        let mut input_sizes: Vec<i64> = vec![];
        let mut output_sizes: Vec<i64> = vec![];
        for lay in (&nn.layers).into_iter() {
            match lay {
                Layers::Linear {
                    name: _name,
                    input_size,
                    output_size,
                    has_biases,
                    weights,
                    biases,
                } => {
                    input_sizes.push(*input_size);
                    output_sizes.push(*output_size);
                    let mut layer_name = "layer".to_owned();
                    layer_name.push_str(&_name);

                    let mut weights_tensor =
                        Tensor::empty(&[weights.len() as i64, weights[0].len() as i64], DOUBLE_CPU);
                    for w in weights {
                        weights_tensor = weights_tensor.f_add(&Tensor::of_slice(&w)).unwrap();
                    }
                    let biases_tensor: Option<Tensor>;
                    if *has_biases {
                        biases_tensor = Some(Tensor::of_slice(&biases));
                    } else {
                        biases_tensor = None;
                    }
                    let linear_layer = Linear {
                        ws: weights_tensor,
                        bs: biases_tensor,
                    };
                    tch_nn = tch_nn.add(linear_layer);
                }
                Layers::ReLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.relu()),
            }
        }
        //println!("{:#?}", tch_nn);
        ModelWrapper {
            _model: tch_nn,
            _nn: nn,
        }
    }
}

impl Clone for ModelWrapper {
    fn clone(&self) -> Self {
        ModelWrapper::new(self._nn.clone())
    }
}

#[derive(Clone)]
pub struct NnOracle<T>
where
    T: time::Time,
{
    //model: Sequential,
    model_wrapper: ModelWrapper,
    _input_size: usize,
    output_size: usize,
    explorer: Arc<Explorer<T>>,
    action_resolver: EdgeByIndexResolver<T>,
}

impl<'a, T> NnOracle<T>
where
    T: time::Time,
{
    pub fn build(
        nn: NeuralNetwork,
        explorer: Arc<Explorer<T>>,
        instance_name: Option<String>,
    ) -> Self {
        let input_size = nn.get_input_size() as usize;
        let output_size = nn.get_output_size() as usize;
        let action_resolver = EdgeByIndexResolver::new(explorer.clone(), instance_name);
        NnOracle {
            _input_size: input_size,
            output_size,
            explorer,
            action_resolver,
            model_wrapper: ModelWrapper::new(Arc::new(nn)),
        }
    }

    fn state_to_tensor(&self, state: &State<T>) -> Tensor {
        let mut vec_values = vec![];
        for (id, _) in &self.explorer.network.declarations.global_variables {
            if id.starts_with("local_") {
                // For the moment, we only use the global.
                continue;
            }
            let g_value = state.get_global_value(&self.explorer, &id).unwrap();
            //println!("({:?},{:?})", id, g_value);
            match g_value.get_type() {
                model::Type::Bool => panic!("Tensor type not valid"),
                model::Type::Vector { element_type: _ } => panic!("Tensor type not valid"),
                model::Type::Unknown => panic!("Tensor type not valid"),
                model::Type::Float64 => vec_values.push(g_value.unwrap_float64().into_inner()),
                model::Type::Int64 => vec_values.push(g_value.unwrap_int64() as f64),
            };
        }
        let tensor = Tensor::of_slice(&vec_values);
        tensor
    }

    fn _tensor_to_action(&self, tensor: Tensor) -> i64 {
        if tensor.size()[0] as usize != self.output_size {
            panic!("Vector size and NN output size does not match");
        }
        let idx = tensor.argmax(None, true).int64_value(&[0]);
        idx
    }
}

impl<T> Oracle<T> for NnOracle<T>
where
    T: time::Time,
{
    fn choose<'s, 't>(
        &self,
        state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        if transitions.len() == 1 {
            transitions.into_iter().next().unwrap()
        } else {
            let mut rng = rand::thread_rng();
            let tensor = self.state_to_tensor(state);

            let output_tensor = self.model_wrapper._model.forward(&tensor);
            let mut tensor_map: HashMap<i64, f64> = HashMap::new();
            for a in 0..self.output_size as i64 {
                tensor_map.insert(a, output_tensor.double_value(&[a]));
            }
            //------------------------------\\
            // let mut action_map: HashMap<i64, f64> = HashMap::new();
            // for a in self.get_edges_ids(transitions).into_iter() {
            //     action_map.insert(a, output_tensor.double_value(&[a]));
            // }
            // let _action = self.tensor_to_action(output_tensor);
            // let selected_transitions = self.action_resolver.resolve_v0(&transitions, max_key);
            //------------------------------\\
            let selected_transitions = self.action_resolver.resolve(&transitions, &tensor_map);
            if selected_transitions.is_empty() {
                return transitions.into_iter().choose(&mut rng).unwrap();
            }
            if selected_transitions.len() > 1 {
                println!("Uncontrolled nondeterminism resolved uniformly.");
            }

            let transition = selected_transitions.into_iter().choose(&mut rng).unwrap();
            transition
        }
    }
}
