use std::{collections::HashMap, sync::Arc};

use serde::Deserialize;
use thiserror::Error;

use crate::ts::{Ts, TsBuilder};

#[derive(Debug, Clone, Deserialize)]
struct YamlTs {
    states: HashMap<Arc<str>, YamlStateData>,
    transitions: Vec<YamlTransition>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YamlStateData {
    is_initial: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct YamlTransition {
    source: Arc<str>,
    target: Arc<str>,
    action: Option<Arc<str>>,
}

/// A parsing error.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct Error(#[from] ErrorInner);

/// Inner error type.
#[derive(Debug, Error)]
enum ErrorInner {
    /// A YAML parsing error.
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
}

/// Constructs an FTS from the provided string.
pub fn from_str(yaml: &str) -> Result<Ts<Arc<str>, Option<Arc<str>>>, Error> {
    serde_yaml::from_str(yaml)
        .map_err(ErrorInner::Yaml)
        .and_then(|yaml_ts: YamlTs| {
            let mut builder = TsBuilder::new();
            for (state_name, state_data) in yaml_ts.states {
                let state_id = builder.insert_state(state_name.clone());
                if state_data.is_initial.unwrap_or_default() {
                    builder.mark_initial(state_id);
                }
            }
            for transition in yaml_ts.transitions {
                let source_id = builder.insert_state(transition.source);
                let target_id = builder.insert_state(transition.target);
                builder.insert_transition(source_id, transition.action, target_id);
            }
            Ok(builder.build())
        })
        .map_err(Into::into)
}
