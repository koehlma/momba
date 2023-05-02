use std::sync::Arc;

use hashbrown::HashMap;
//use std::sync::Arc;
use momba_explore::*;
use rand::seq::IteratorRandom;
use rayon::vec;
use serde::{Deserialize, Serialize};
use tch::{
    kind::*,
    nn,
    nn::{Linear, Module, Sequential},
    Device, Tensor,
};

use crate::{
    nn_oracle::generic::{Context, GenericExplorer},
    simulate::{Oracle, Simulator},
};

use self::generic::{ActionResolver, EdgeByIndexResolver};

mod generic;

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layers>,
}

#[derive(Debug, Deserialize, Serialize)]
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

pub struct NnSimulator<'a, T, G>
//pub struct NnSimulator<'a, T, G, A>
where
    T: time::Time,
    //A: ActionResolver<T>,
{
    gen_exp: GenericExplorer<'a, T, G>,
    input_size: usize,
    model: Sequential,
}
impl<'a, T, G> NnSimulator<'a, T, G>
//impl<'a, T, G, A> NnSimulator<'a, T, G, A>
where
    T: time::Time,
    G: Fn(&State<T>) -> bool,
    //A: ActionResolver<T>,
{
    pub fn _new(model: Sequential, exp: &'a Explorer<T>, goal: G, size: usize) -> Self {
        //let _test = generic::EdgeByLabelResolver::new(exp);
        let ctx = Context::new(
            exp,
            generic::Actions::EdgeByIndex,
            generic::Observations::GlobalOnly,
        );
        NnSimulator {
            model,
            gen_exp: GenericExplorer::new(ctx, goal),
            input_size: size,
        }
    }
    pub fn fn1(&self, state: &State<T>, size: usize) -> Tensor {
        self.gen_exp.state_to_tensor(state, size)
    }
    //IDEA: Should change the state of the explorator (by calling step).
    //          Can a reference to the new state so it satisfies the
    //          next function of the simulation trait
    //      Or only get the tensor and use the action resolve to get
    //      the transition vector and then uniform choose one?
    pub fn fn2(&mut self, tensor: Tensor) {
        //-> Transition<'static, T>{
        let action = self.gen_exp.tensor_to_action(tensor);
        let _output = self.gen_exp.step(action);
        //println!("Action: {:?}. Output: {:?}", action, _output);
    }
}

impl<T, G> Simulator for NnSimulator<'_, T, G>
//impl<T, G, A> Simulator for NnSimulator<'_, T, G, A>
where
    T: time::Time,
    G: Fn(&State<T>) -> bool,
    //A: ActionResolver<T>,
{
    type State<'sim> = &'sim State<T> where Self:'sim;
    fn current_state(&mut self) -> Self::State<'_> {
        &self.gen_exp.state
    }

    /*If I dont need to resolve any undetermination, then avoid calling the NN.*/
    fn next(&mut self) -> Option<Self::State<'_>> {
        let new_state: &State<T>;
        if self.gen_exp.has_choice() {
            let in_tensor = self.fn1(&self.gen_exp.state, self.input_size);
            let out_tensor = self.model.forward(&in_tensor);
            self.fn2(out_tensor);
            new_state = &self.gen_exp.state;
        } else {
            let _output = self.gen_exp.uniform_step();
            //println!("Uniform Step. Output: {:?}", _output);
            new_state = &self.gen_exp.state;
        }
        Some(new_state)
    }

    fn reset(&mut self) -> Self::State<'_> {
        self.gen_exp.reset();
        &self.gen_exp.state
    }
}

//pub fn build_nn(nn: NeuralNetwork) -> (impl Module, usize) {
pub fn _build_nn(nn: NeuralNetwork) -> (Sequential, usize) {
    let mut tch_nn = nn::seq();
    let _vs = nn::VarStore::new(Device::Cpu);
    let mut input_sizes: Vec<i64> = vec![];
    for lay in nn.layers.into_iter() {
        match lay {
            Layers::Linear {
                name,
                input_size,
                output_size: _,
                has_biases,
                weights,
                biases,
            } => {
                input_sizes.push(input_size);
                let mut layer_name = "layer".to_owned();
                layer_name.push_str(&name);

                let mut weights_tensor =
                    Tensor::empty(&[weights.len() as i64, weights[0].len() as i64], DOUBLE_CPU);
                for w in weights {
                    weights_tensor = weights_tensor.f_add(&Tensor::of_slice(&w)).unwrap();
                }
                let biases_tensor: Option<Tensor>;
                if has_biases {
                    biases_tensor = Some(Tensor::of_slice(&biases));
                } else {
                    biases_tensor = None;
                }
                let linear_layer = Linear {
                    ws: weights_tensor,
                    bs: biases_tensor,
                };
                tch_nn = tch_nn.add(linear_layer);

                // If its a new model, it should be initialized this way, and then be trained.
                // tch_nn = tch_nn.add(nn::linear(
                //     &vs.root() / layer_name,
                //     input_size,
                //     output_size,
                //     //LinearConfig { ws_init: (weights), bs_init: (biases), bias: has_biases }
                //     Default::default(),
                // ));
            }
            Layers::ReLU { name: _ } => tch_nn = tch_nn.add_fn(|xs| xs.relu()),
        }
    }
    (tch_nn, input_sizes[0] as usize)
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

// This you can clone the neural network for each thread.
struct ModelWrapper {
    _model: Sequential,
    _og: Arc<NeuralNetwork>, //Change the name pls
}

impl Clone for ModelWrapper {
    fn clone(&self) -> Self {
        todo!()
    }
}

//#[derive(Clone)] //If its really needed for the oracle trait, then its a problem
// beacuse sequential does not implement Clone trait.
pub struct NnOracle<T>
where
    T: time::Time,
{
    model: Sequential,
    input_size: usize,
    output_size: usize,
    //it should'nt have the explorer.
    //explorer: &'a Explorer<T>,
    explorer: Arc<Explorer<T>>,
    action_resolver: EdgeByIndexResolver<T>,
}

impl<'a, T> NnOracle<T>
where
    T: time::Time,
{
    //pub fn build(nn: NeuralNetwork, explorer: &'a Explorer<T>) -> Self {
    pub fn build(nn: NeuralNetwork, explorer: Arc<Explorer<T>>) -> Self {
        let mut tch_nn = nn::seq();
        let mut default_nn = nn::seq();
        let _vs = nn::VarStore::new(Device::Cpu);
        let mut input_sizes: Vec<i64> = vec![];
        let mut output_sizes: Vec<i64> = vec![];
        for lay in nn.layers.into_iter() {
            match lay {
                Layers::Linear {
                    name: _name,
                    input_size,
                    output_size,
                    has_biases,
                    weights,
                    biases,
                } => {
                    input_sizes.push(input_size);
                    output_sizes.push(output_size);
                    let mut layer_name = "layer".to_owned();
                    layer_name.push_str(&_name);

                    // Find a way to know the type of the tensor values.
                    let mut weights_tensor =
                        Tensor::empty(&[weights.len() as i64, weights[0].len() as i64], DOUBLE_CPU);
                    // Here it should be double, because the tch tool takes the f64 as doubles for
                    // each of the vectors.
                    for w in weights {
                        //println!("LALA:");
                        //Tensor::of_slice(&w).print();
                        weights_tensor = weights_tensor.f_add(&Tensor::of_slice(&w)).unwrap();
                    }
                    //weights_tensor.print();

                    let biases_tensor: Option<Tensor>;
                    if has_biases {
                        biases_tensor = Some(Tensor::of_slice(&biases));
                    } else {
                        biases_tensor = None;
                    }
                    let linear_layer = Linear {
                        ws: weights_tensor,
                        bs: biases_tensor,
                    };
                    tch_nn = tch_nn.add(linear_layer);
                    default_nn = default_nn.add(nn::linear(
                        &_vs.root() / layer_name,
                        input_size,
                        output_size,
                        Default::default(),
                    ));
                }
                Layers::ReLU { name: _ } => {
                    tch_nn = tch_nn.add_fn(|xs| xs.relu());
                    default_nn = default_nn.add_fn(|xs| xs.relu())
                }
            }
        }
        //println!("{:#?}\n\nvs\n\n", tch_nn);
        //println!("{:#?}", default_nn);

        let action_resolver = EdgeByIndexResolver::new(explorer.clone());
        NnOracle {
            model: tch_nn,
            input_size: input_sizes.into_iter().next().unwrap() as usize,
            output_size: output_sizes.into_iter().last().unwrap() as usize,
            explorer,
            //explorer: Arc::new(explorer),
            action_resolver,
        }
    }

    fn state_to_tensor(&self, state: &State<T>) -> Tensor {
        //Old version
        // let mut tensor = Tensor::empty(&[self.input_size as i64], INT64_CPU);
        // for (id, _) in &self.explorer.network.declarations.global_variables {
        //     let g_value = state.get_global_value(&self.explorer, &id).unwrap();
        //     tensor = match g_value.get_type() {
        //         model::Type::Bool => panic!("Tensor type not valid"),
        //         //tensor.f_add_scalar(g_value.unwrap_bool()).unwrap(),
        //         model::Type::Float64 => tensor
        //             .f_add_scalar(g_value.unwrap_float64().into_inner())
        //             .unwrap(),
        //         model::Type::Int64 => tensor.f_add_scalar(g_value.unwrap_int64()).unwrap(),
        //         model::Type::Vector { element_type: _ } => panic!("Tensor type not valid"),
        //         //tensor.f_add_(g_value.unwrap_vector()).unwrap(),
        //         model::Type::Unknown => panic!("Tensor type not valid"),
        //     };
        //     tensor.print();
        // }
        // for (id, _) in &self.explorer.network.declarations.transient_variables {
        //     let l_value = state
        //         .get_transient_value(&self.explorer.network, id)
        //         .unwrap_int64();
        //     tensor = tensor.f_add_scalar(l_value).unwrap();
        // }
        // tensor
        let mut vec_values = vec![];
        for (id, v) in &self.explorer.network.declarations.global_variables {
            let g_value = state.get_global_value(&self.explorer, &id).unwrap();
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

    // The idea is to have another function that will return me the available action with the highest
    // q value from the model.
    fn get_edges_ids(&self, transitions: &[Transition<T>]) -> Vec<i64> {
        //Should be a filter and a flatten.
        let mut actions = vec![];
        for t in transitions.into_iter() {
            for (ins, val) in t.numeric_reference_vector().into_iter() {
                match ins {
                    0 => actions.push(val as i64),
                    _ => continue,
                }
            }
        }
        actions
    }

    fn tensor_to_action(&self, tensor: Tensor) -> i64 {
        if tensor.size()[0] as usize != self.output_size {
            panic!("Vector size and NN output size does not match");
        }
        let idx = tensor.argmax(None, true).int64_value(&[0]);
        //println!(
        //    "Max value overall {:?} in action:{:?}",
        //    tensor.double_value(&[idx]),
        //    idx
        //);

        idx
    }
}

impl<T> Oracle<T> for NnOracle<T>
where
    T: time::Time,
{
    /*

    */
    fn choose<'s, 't>(
        &self,
        state: &State<T>,
        transitions: &'t [Transition<'s, T>],
    ) -> &'t Transition<'t, T> {
        //println!("Transitions Received in choose: {:?}", transitions.len());
        if transitions.len() == 1 {
            transitions.into_iter().next().unwrap()
        } else {
            let mut rng = rand::thread_rng();
            let tensor = self.state_to_tensor(state);

            let output_tensor = self.model.forward(&tensor);

            let mut action_map: HashMap<i64, f64> = HashMap::new();
            //let mut available_transitions = vec![];
            //let mut out = vec![];
            //self.action_resolver.available(state, &mut out);
            //for (i, t) in transitions.iter().enumerate() {
            //    if out[i] {
            //        available_transitions.push(t)
            //    }
            //}
            //println!();
            //println!("{:?}", out);
            for a in self.get_edges_ids(transitions).into_iter() {
                action_map.insert(a, output_tensor.double_value(&[a]));
            }
            let mut max_val: f64 = 0.0;
            let mut max_key: i64 = -1;
            for (k, v) in action_map.iter() {
                if *v > max_val {
                    max_key = *k;
                    max_val = *v;
                }
            }

            let _action = self.tensor_to_action(output_tensor);
            let selected_transitions = self.action_resolver.resolve(&transitions, max_key);

            //let action = self.tensor_to_action(output_tensor);
            //let selected_transitions = self.action_resolver.resolve(&transitions, action);

            if selected_transitions.is_empty() {
                //println!(
                //    "he choose poorly: {:?}. Resolving uniformly between {:?} actions",
                //    action,
                //    transitions.len()
                //);
                return transitions.into_iter().choose(&mut rng).unwrap();
            } else {
                //println!("You have choosen wisely: {:?}", action);
            }
            if selected_transitions.len() > 1 {
                println!("Uncontrolled nondeterminism resolved uniformly.");
            }

            let transition = selected_transitions.into_iter().choose(&mut rng).unwrap();
            transition
        }
    }
}
