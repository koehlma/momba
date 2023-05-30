use std::{cell::RefCell, sync::Arc};

use hashbrown::HashMap;
use momba_explore::*;
use rand::{rngs::StdRng, seq::IteratorRandom, Rng};
use serde::{Deserialize, Serialize};
use tch::{
    kind::*,
    nn,
    nn::{Linear, Module, Sequential},
    Device, Tensor,
};

use crate::simulate::Oracle;

use self::generic::{ActionResolver, EdgeByIndexResolver};

mod generic;

/// Structure that allows the reading of the Neural Network from a json file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JsonNN {
    /// Each layer of the model.
    layers: Vec<Layers>,
}

/// This allows us to read different types of layers in the model.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "kind")]
enum Layers {
    /// Linear layer.
    #[serde(rename_all = "camelCase")]
    Linear {
        name: String,
        input_size: i64,
        output_size: i64,
        has_biases: bool,
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
    },
    //> Biliear layer.
    #[serde(rename_all = "camelCase")]
    Bilinear {
        name: String,
        input_size1: i64,
        input_size2: i64,
        output_size: i64,
        has_biases: bool,
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
    },
    /// Activation Functions
    ReLU {
        name: String,
    },
    CeLU {
        name: String,
    },
    ELU {
        name: String,
    },
    GeLU {
        name: String,
    },
    LeakyReLU {
        name: String,
    },
    Mish {
        name: String,
    },
    RReLU {
        name: String,
    },
    Sigmod {
        name: String,
    },
    LogSoftmax {
        name: String,
    },
    Softmax {
        name: String,
    },
    Tanh {
        name: String,
    },
}

/// Implementation of the struct for reading the json files.
impl JsonNN {
    fn new(layers: Vec<Layers>) -> Self {
        JsonNN { layers }
    }
    /// Outputs the input size of the model.
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
    /// Outputs the output size of the model.
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

pub fn _default_nn(input_size: i64, output_size: i64, number_of_layers: u32) -> ModelWrapper {
    let mut rng = rand::thread_rng();
    let mut layers: Vec<Layers> = vec![];
    for i in 0..number_of_layers {
        let mut in_size = input_size;
        let mut out_size = output_size;
        if i != 0 {
            in_size = 64;
        }
        if i != (number_of_layers - 1) {
            out_size = 64;
        }
        //let test: Vec<f64> = (0..in_size).map(|v| (v / in_size) as f64).collect();
        let test: Vec<Vec<f64>> = (0..in_size)
            .map(|_| (0..out_size).map(|_| rng.gen::<f64>()).collect())
            .collect();
        let layer = Layers::Linear {
            name: i.to_string(),
            input_size: in_size,
            output_size: out_size,
            has_biases: false,
            weights: test,
            biases: vec![],
        };
        let relu = Layers::ReLU {
            name: (i + 1).to_string(),
        };
        layers.push(layer);
        if i != (number_of_layers - 1) {
            layers.push(relu);
        }
    }
    let json_nn = JsonNN::new(layers);
    ModelWrapper::new(Arc::new(json_nn))
}

/// Wrapper for the models, that will allow us to create the model and to
/// clone and not lose the reference to the original JsonNN so we can create
/// different instances for the parallel simulations.
pub struct ModelWrapper {
    pub model: Sequential,
    pub nn: Arc<JsonNN>,
}
/// Implementation of the wrapper.
impl ModelWrapper {
    fn new(nn: Arc<JsonNN>) -> Self {
        let _vs = nn::VarStore::new(Device::Cpu);
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
                        Tensor::zeros(&[weights.len() as i64, weights[0].len() as i64], DOUBLE_CPU);

                    for (i, w) in weights.iter().enumerate() {
                        let tensor = Tensor::of_slice(&w);
                        let idx = Tensor::of_slice(&[i as i32]);
                        weights_tensor = weights_tensor.index_put(&[Some(idx)], &tensor, true);
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
                Layers::Bilinear {
                    name: _,
                    input_size1: _,
                    input_size2: _,
                    output_size: _,
                    has_biases: _,
                    weights: _,
                    biases: _,
                } => {
                    panic!("BiLinear layer not yet supported!")
                }
                Layers::ReLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.relu()),
                Layers::CeLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.celu()),
                Layers::ELU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.elu()),
                Layers::GeLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.gelu("none")),
                Layers::LeakyReLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.leaky_relu()),
                Layers::Mish { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.mish()),
                Layers::RReLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.rrelu(false)),
                Layers::Sigmod { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.sigmoid()),
                Layers::LogSoftmax { name: _ } => {
                    tch_nn = tch_nn.add_fn(|xs| xs.log_softmax(0, Kind::Float))
                }
                Layers::Softmax { name: _ } => {
                    tch_nn = tch_nn.add_fn(|xs| xs.softmax(0, Kind::Float))
                }
                Layers::Tanh { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.tanh()),
            }
        }
        //println!("{:#?}", tch_nn);
        ModelWrapper { model: tch_nn, nn }
    }
}

/// Cloning capabilities for the model.
impl Clone for ModelWrapper {
    fn clone(&self) -> Self {
        ModelWrapper::new(self.nn.clone())
    }
}

/// Structure that will represent the Oracle that will use the NN.
#[derive(Clone)]
pub struct NnOracle<T>
where
    T: time::Time,
{
    /// Wrapper for the network.
    model_wrapper: ModelWrapper,
    /// Input size of the model.
    _input_size: usize,
    /// Output size of the model
    output_size: usize,
    /// Explorer reference to the oracle.
    explorer: Arc<Explorer<T>>,
    /// Action resolver that helps in resolving the un-determinations.
    action_resolver: EdgeByIndexResolver<T>,
    /// RNG for the oracle.
    rng: RefCell<StdRng>,
}

impl<'a, T> NnOracle<T>
where
    T: time::Time,
{
    /// Construct the NN Oracle from the Json representation of the NN, a reference to the explorer
    /// the instance name of the controlled automaton, and a seeded rng.
    pub fn build(
        nn: JsonNN,
        explorer: Arc<Explorer<T>>,
        instance_name: Option<String>,
        rng: StdRng,
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
            rng: RefCell::new(rng),
        }
    }

    fn state_to_tensor(&self, state: &State<T>) -> Tensor {
        let mut vec_values = vec![];
        for (id, t) in &self.explorer.network.declarations.global_variables {
            if id.starts_with("local_") {
                // For the moment, we only use the global.
                // TODO: check what to do about the observations, global, local and omnicient.
                continue;
            }
            match t {
                model::Type::Bool => panic!("Tensor type not valid"),
                model::Type::Vector { element_type: _ } => panic!("Tensor type not valid"),
                model::Type::Unknown => panic!("Tensor type not valid"),
                model::Type::Float64 => vec_values.push(
                    state
                        .get_global_value(&self.explorer, &id)
                        .unwrap()
                        .unwrap_float64()
                        .into_inner(),
                ),
                model::Type::Int64 => vec_values.push(
                    state
                        .get_global_value(&self.explorer, &id)
                        .unwrap()
                        .unwrap_int64() as f64,
                ),
            };
        }
        let tensor = Tensor::of_slice(&vec_values);
        tensor
    }

    /// This function only makes an argmax of the tensor, and returns the action with the highest value
    /// We dont use it, because maybe the highest value action is not available on the state you are.
    /// This can happen because of the NN is a function and not a map.
    fn _greedy_tensor_to_action(&self, tensor: Tensor) -> i64 {
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
    /// From a state and a group of transitions, choosese the next one using the NN as an oracle.
    /// The NN outputs the aproximated Q_values for each transitions, then we
    /// clear this transitions using the action resolver.
    /// Panics if the NN outputs NaN or infs.
    fn choose<'s, 't>(
        &self,
        state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        let mut rng = self.rng.borrow_mut();
        let mut out: Vec<bool> = vec![];
        self.action_resolver.available_v0(state, &mut out);
        if !out.into_iter().any(|b| b) {
            // If there isn't an available action from the controlled instance.
            // choose uniformly from the transitions.
            return transitions.into_iter().choose(&mut *rng).unwrap();
        } else {
            if transitions.len() == 1 {
                return transitions.first().unwrap();
            } else {
                let tensor = self.state_to_tensor(state);
                let output_tensor = self.model_wrapper.model.forward(&tensor);

                let mut tensor_map: HashMap<i64, f64> = HashMap::new();
                for i in 0..self.output_size as i64 {
                    let value = output_tensor.double_value(&[i]);
                    if value.is_nan() {
                        panic!("The NN returned values that are NAN. Was correctly trained?")
                    };
                    tensor_map.insert(i, value);
                }

                let selected_transitions = self.action_resolver.resolve(&transitions, &tensor_map);
                //println!(
                //    "Len Transitions: {:?}- Len Sel Transitions: {:?}",
                //    transitions.len(),
                //    selected_transitions.len()
                //);
                if selected_transitions.is_empty() {
                    panic!("Empty selected transitions...");
                    //return transitions.into_iter().choose(&mut *rng).unwrap();
                }
                return selected_transitions.into_iter().choose(&mut *rng).unwrap();
            }
        }

        // let tensor = self.state_to_tensor(state);
        // let output_tensor = self.model_wrapper.model.forward(&tensor);

        // let mut tensor_map: HashMap<i64, f64> = HashMap::new();
        // for i in 0..self.output_size as i64 {
        //     let value = output_tensor.double_value(&[i]);
        //     if value.is_nan() {
        //         panic!("The NN returned values that are NAN. Was correctly trained?")
        //     };
        //     tensor_map.insert(i, value);
        // }

        // //println!("{:#?}", tensor_map);

        // let selected_transitions = self.action_resolver.resolve(&transitions, &tensor_map);
        // if selected_transitions.is_empty() {
        //     panic!("Empty selected transitions...");
        //     //return transitions.into_iter().choose(&mut *rng).unwrap();
        // }
        // if selected_transitions.len() > 1 {
        //     println!("Uncontrolled nondeterminism resolved uniformly.");
        // }

        // let transition = selected_transitions.into_iter().choose(&mut *rng).unwrap();
        // println!("---");
        // transition
        //}
    }
}
