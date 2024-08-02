//! Output in [Pseuco's](https://pseuco.com) LTS JSON format.

use std::{collections::HashMap, fs, hash::Hash, io, path::Path};

use serde::{de, Deserialize, Serialize};

use crate::ts::{
    traits::{BaseTs, InitialStates, States},
    Ts,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct State {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transitions: Vec<Transition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlighted: Option<bool>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Transition {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weak: Option<bool>,
    pub details_label: DetailsLabel,
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct DetailsLabel(pub Option<String>);

impl Serialize for DetailsLabel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if let Some(label) = &self.0 {
            serializer.serialize_str(label)
        } else {
            serializer.serialize_bool(false)
        }
    }
}

impl<'de> Deserialize<'de> for DetailsLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visitor;

        impl<'de> de::Visitor<'de> for Visitor {
            type Value = DetailsLabel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("Expected a string or `false`.")
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if !v {
                    Ok(DetailsLabel(None))
                } else {
                    Err(de::Error::invalid_type(de::Unexpected::Bool(v), &self))
                }
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DetailsLabel(Some(v.to_owned())))
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PseucoLts {
    initial_state: String,
    states: HashMap<String, State>,
}

pub fn to_pseuco_lts<S, L>(
    ts: &Ts<S, L>,
    state_label: impl Fn(&S) -> String,
    transition_label: impl Fn(&L) -> String,
) -> PseucoLts
where
    S: Clone + Eq + Hash,
    L: Clone + Eq + Hash,
{
    let mut state_ids = HashMap::new();
    for state in ts.states() {
        let state_id = format!(
            "{} ({})",
            state_label(ts.get_label(&state)),
            state_ids.len()
        );
        state_ids.insert(state, state_id);
    }

    let initial_state = ts
        .initial_states()
        .map(|initial_state| state_ids.get(&initial_state))
        .next()
        .unwrap()
        .unwrap()
        .clone();

    PseucoLts {
        initial_state,
        states: ts
            .states()
            .map(|state| {
                (
                    state_ids.get(&state).unwrap().clone(),
                    State {
                        transitions: ts
                            .outgoing(&state)
                            .map(|transition| Transition {
                                label: transition_label(&transition.label),
                                weak: Some(false),
                                details_label: DetailsLabel(None),
                                target: state_ids.get(&transition.target).unwrap().clone(),
                            })
                            .collect(),
                        error: None,
                        highlighted: None,
                    },
                )
            })
            .collect(),
    }
}

pub fn write_to_file<S, L>(
    lts: &Ts<S, L>,
    path: &Path,
    state_label: impl Fn(&S) -> String,
    transition_label: impl Fn(&L) -> String,
) -> Result<(), io::Error>
where
    S: Clone + Eq + Hash,
    L: Clone + Eq + Hash,
{
    let src = serde_json::to_string(&to_pseuco_lts(lts, state_label, transition_label)).unwrap();
    fs::write(path, &src)
}
