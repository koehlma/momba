use momba_explore::*;
use serde::{Deserialize, Serialize};
use tch::{
    kind::*,
    nn,
    nn::{Linear, Module, Sequential},
    Device, Tensor,
};

use crate::{
    nn_oracle::generic::{Context, GenericExplorer},
    simulate::{Simulator, SimulationOutput},
};

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
    pub fn new(model: Sequential, exp: &'a Explorer<T>, goal: G, size: usize) -> Self {
        //pub fn new(exp: &'a Explorer<T>, actions:Actions) -> Self {
        //let _test = generic::EdgeByLabelResolver::new(exp);
        let ctx = Context::new(exp, generic::Actions::EdgeByIndex);
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
    //          next funciton of the simulation trait
    //       Or only get the tensor and use the action resolve to get
    //       the transition vector and then uniform choose one?
    pub fn fn2(&mut self, tensor: Tensor) {
        //-> Transition<'static, T>{
        let action = self.gen_exp.tensor_to_action(tensor);
        let _output = self.gen_exp.step(action);
        //println!("Action: {:?}. Output: {:?}", action, _output);
    }

    pub fn simulate(&mut self)-> SimulationOutput{
        self.reset();
        let mut c = 0;
        while let Some(_state) = self.next() {
            //let next_state = state.into();
            if self.gen_exp.has_reached_goal() {
                return SimulationOutput::GoalReached;
            } else if c >= 99 {
                return SimulationOutput::MaxSteps;
            }
            c += 1;
        }
        return SimulationOutput::NoStatesAvailable;
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

/*
fn _state_to_tensor_old<T: time::Time>(
    state: &State<T>,
    explorer: &Explorer<T>,
    size: usize,
) -> Tensor {
    let input_size = size;
    let mut tensor = Tensor::empty(&[input_size as i64], DOUBLE_CPU);
    //for the moment, only supports tneosrs of floats or Ints.
    for id in &explorer.network.declarations.global_variables {
        let g_value = state.get_global_value(&explorer, &id.0).unwrap();
        tensor = match g_value.get_type() {
            model::Type::Bool => panic!("Tensor type not valid"), //tensor.f_add_scalar(g_value.unwrap_bool()).unwrap(),
            model::Type::Float64 => tensor
                .f_add_scalar(g_value.unwrap_float64().into_inner())
                .unwrap(),
            model::Type::Int64 => tensor.f_add_scalar(g_value.unwrap_int64()).unwrap(),
            model::Type::Vector { element_type: _ } => panic!("Tensor type not valid"), //tensor.f_add_(g_value.unwrap_vector()).unwrap(),
            model::Type::Unknown => panic!("Tensor type not valid"),
        };
        //tensor = tensor.f_add_scalar(g_value).unwrap();
    }
    for id in &explorer.network.declarations.transient_variables {
        let l_value = state
            .get_transient_value(&explorer.network, id.0)
            .unwrap_int64();
        tensor = tensor.f_add_scalar(l_value).unwrap();
    }
    if size != tensor.size()[0] as usize {
        panic!("Vector size and NN input size does not match")
    };
    tensor
}

fn _tensor_to_transition_old<T: time::Time>(
    explorer: &Explorer<T>,
    tensor: Tensor,
) -> Option<Transition<'static, T>> {
    let action_resolver = EdgeByIndexResolver::new(explorer);
    let ctx = Context::new(&explorer, generic::Actions::EdgeByIndex);

    let gen_exp = GenericExplorer::new(ctx);

    let init_state = explorer.initial_states().into_iter().next().unwrap();
    let tran = explorer
        .transitions(&init_state)
        .into_iter()
        .next()
        .unwrap();
    let dest = explorer
        .destinations(&init_state, &tran)
        .into_iter()
        .next()
        .unwrap();
    let snd_state = explorer.successor(&init_state, &tran, &dest);
    //println!("#Transitions: {:?}", explorer.transitions(&snd_state).len());
    //let mut out_vec: Vec<bool> = vec![];
    //nose.available(snd_state, &mut out_vec);
    //println!("{:?}", out_vec);

    let mut all_transitions = explorer.transitions(&snd_state);
    println!("Len: {:?}", all_transitions.len());
    let nosep = action_resolver.resolve(&mut all_transitions, 16);
    println!("Len: {:?}", nosep.len());

    // The true/false is the same if the dim is None.
    //println!("Index of action choosed with argmax: {:?}", tensor.argmax(None, true));
    //tensor.print();

    //println!("{:?}",nosep.locations[0].visible_edges[0].);
    //println!("PRINT:{:?}", explorer.network.get_edge());
    None
}
*/

//pub fn build_nn(nn: NeuralNetwork) -> (impl Module, usize) {
pub fn build_nn(nn: NeuralNetwork) -> (Sequential, usize) {
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
