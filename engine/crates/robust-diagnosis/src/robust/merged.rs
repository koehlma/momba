use momba_explore::model;

type Explorer = momba_explore::Explorer<momba_explore::time::Float64Zone>;
type State = momba_explore::State<momba_explore::time::Float64Zone>;

pub struct Node {
    pub(crate) state: State,
}

pub struct Graph {
    pub(crate) explorer: Explorer,
}

impl Graph {
    pub(crate) fn new(network: model::Network) -> Self {
        Self {
            explorer: momba_explore::Explorer::new(network),
        }
    }
}
