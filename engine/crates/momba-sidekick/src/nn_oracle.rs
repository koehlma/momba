use std::sync::Arc;

use hashbrown::HashMap;
use momba_explore::*;
use rand::seq::IteratorRandom;
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
    // There can be much more different types of NN
    /// Linear Models.
    #[serde(rename_all = "camelCase")]
    Linear {
        name: String,
        input_size: i64,
        output_size: i64,
        has_biases: bool,
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
    },
    /// Rectified Linear Unit (ReLU) is a layer that translates to a
    /// non-linear activation function.
    ReLU { name: String },
}

/// Implementation of the struct for reading the json files.
impl JsonNN {
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

/// Wrapper for the models, that will allow us to create the model and to
/// clone and not lose the reference to the original JsonNN so we can create
/// different instances for the parallel simulations.
struct ModelWrapper {
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
                        //Tensor::empty(&[weights.len() as i64, weights[0].len() as i64], FLOAT_CPU);
                        Tensor::zeros(&[weights.len() as i64, weights[0].len() as i64], FLOAT_CPU);

                    for (i, w) in weights.iter().enumerate() {
                        let tensor = Tensor::of_slice(&w).to_dtype(Kind::Float, true, false);
                        let idx = Tensor::of_slice(&[i as i32]);

                        // weights_tensor.print();
                        // HERE WAS THE ERROR!!!!! It can be seen how its actaully adding the values, and that its not desired at all.
                        // weights_tensor = weights_tensor.f_add(&Tensor::of_slice(&w)).unwrap();

                        //weights_tensor.index_fill_int_tensor(0, &idx, &tensor);
                        weights_tensor = weights_tensor.index_put(&[Some(idx)], &tensor, true);
                        // index_fill_int_tensor(0, &idx, &tensor);
                        //weights_tensor.print();
                    }

                    let biases_tensor: Option<Tensor>;
                    if *has_biases {
                        let aux = Tensor::of_slice(&biases).to_dtype(Kind::Float, true, false);
                        biases_tensor = Some(aux);
                    } else {
                        biases_tensor = None;
                    }

                    // println!(
                    //     "{:?}.{:?}.{:?}",
                    //     weights_tensor.kind(),
                    //     weights_tensor.size(),
                    //     weights_tensor.dim()
                    // );

                    // weights_tensor.print();

                    let linear_layer = Linear {
                        ws: weights_tensor,
                        bs: biases_tensor,
                    };
                    tch_nn = tch_nn.add(linear_layer);

                    // Default settings for the layer. used if there is intention in training.
                    //tch_nn = tch_nn.add(nn::linear(
                    //    &_vs.root() / layer_name,
                    //    *input_size,
                    //    *output_size,
                    //    //LinearConfig { ws_init: weights_tensor, bs_init: None, bias: has_biases }
                    //    Default::default(),
                    //));
                }
                Layers::ReLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.relu()),
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
    /// Explorer for the
    explorer: Arc<Explorer<T>>,
    ///
    action_resolver: EdgeByIndexResolver<T>,
}

impl<'a, T> NnOracle<T>
where
    T: time::Time,
{
    pub fn build(nn: JsonNN, explorer: Arc<Explorer<T>>, instance_name: Option<String>) -> Self {
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
        //let tensor = Tensor::of_slice(&vec_values);
        let tensor = Tensor::of_slice(&vec_values).to_dtype(Kind::Float, false, false);
        tensor
    }

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
    fn choose<'s, 't>(
        &self,
        state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        if transitions.len() == 1 {
            transitions.into_iter().next().unwrap()
        } else {
            //let mut srng = StdRng::seed_from_u64(42);
            let mut rng = rand::thread_rng();
            let tensor = self.state_to_tensor(state);

            let output_tensor = self.model_wrapper.model.forward(&tensor);
            //println!("Output Tensor: {:?}", output_tensor);
            //output_tensor.print();
            let mut tensor_map: HashMap<i64, f64> = HashMap::new();
            let mut nan_flag = false;

            //println!("Output Tensor: {}", output_tensor);
            //output_tensor.to_dtype(Kind::Float, false, false).print();

            for a in 0..self.output_size as i64 {
                let value = output_tensor.double_value(&[a]);
                nan_flag |= value.is_nan();
                tensor_map.insert(a, value);
            }
            if nan_flag {
                panic!("The NN returned values that are NAN.")
            }
            //if nan_flag {
            //    // Why the NN is returning NaN, what does this means?
            //    //println!("NN returned NaN. Resolving uniformly.");
            //    return transitions.into_iter().choose(&mut rng).unwrap();
            //}

            let selected_transitions = self.action_resolver.resolve(&transitions, &tensor_map);
            if selected_transitions.is_empty() {
                println!("Empty selected transitions...");
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
