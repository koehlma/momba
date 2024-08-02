//! Frontend for the XML-based FTS format used by the
//! [VIBeS project](https://projects.info.unamur.be/vibes/index.html).

use std::{str::FromStr, sync::Arc};

use serde::Deserialize;
use thiserror::Error;

use crate::{
    logic::propositional::Formula,
    ts::{
        types::{Vats, VatsLabel},
        TsBuilder,
    },
};

/// An XML `<fts>` element specifying the FTS.
#[derive(Debug, Clone, Deserialize)]
struct XmlFts {
    /// The initial state of the FTS.
    start: Arc<str>,
    /// The states of the FTS.
    states: XmlStates,
}

/// An XML `<states>` element containing the states of the FTS.
#[derive(Debug, Clone, Deserialize)]
struct XmlStates {
    /// The states.
    #[serde(rename = "state")]
    states: Vec<XmlState>,
}

/// An XML `<state>` element describing a state and its transitions.
#[derive(Debug, Clone, Deserialize)]
struct XmlState {
    /// The id of the state.
    #[serde(rename = "@id")]
    id: Arc<str>,
    /// The transitions of the state.
    #[serde(rename = "transition", default)]
    transitions: Vec<XmlTransition>,
}

/// An XML `<transition>` element describing a transition.
#[derive(Debug, Clone, Deserialize)]
struct XmlTransition {
    /// The optional action of the transition.
    #[serde(rename = "@action")]
    action: Option<Arc<str>>,
    /// The feature guard of the transition.
    #[serde(rename = "@fexpression")]
    guard: Option<Arc<str>>,
    /// The target state of the transition.
    #[serde(rename = "@target")]
    target: Arc<str>,
}

/// A parsing error.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct Error(#[from] ErrorInner);

/// Inner error type.
#[derive(Debug, Error)]
enum ErrorInner {
    /// An XML parsing error.
    #[error(transparent)]
    Xml(#[from] quick_xml::DeError),
    /// Error parsing a feature guard.
    #[error("Error parsing formula {formula}.")]
    Guard {
        formula: Arc<str>,
        error: <Formula<String> as FromStr>::Err,
    },
}

/// Constructs an FTS from the provided string.
pub fn from_str(xml: &str) -> Result<Vats<Arc<str>, Option<Arc<str>>, Formula<String>>, Error> {
    quick_xml::de::from_str(xml)
        .map_err(ErrorInner::Xml)
        .and_then(|xml_fts: XmlFts| {
            let mut builder = TsBuilder::new();
            let initial = builder.insert_state(xml_fts.start.clone());
            builder.mark_initial(initial);
            for xml_state in xml_fts.states.states {
                let source = builder.insert_state(xml_state.id.clone());
                for xml_transition in xml_state.transitions {
                    let guard = xml_transition.guard.map_or_else(
                        || Ok(Formula::True),
                        |guard| {
                            guard.parse().map_err(|error| ErrorInner::Guard {
                                formula: guard.clone(),
                                error,
                            })
                        },
                    )?;
                    let target = builder.insert_state(xml_transition.target.clone());
                    builder.insert_transition(
                        source,
                        VatsLabel::new(xml_transition.action, guard),
                        target,
                    )
                }
            }
            Ok(builder.build())
        })
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    macro_rules! impl_test_fn_for_model {
        ($func:ident, $file:literal) => {
            #[test]
            fn $func() {
                let xml = fs::read_to_string(concat!("../../models/vibes/", $file)).unwrap();
                from_str(&xml).unwrap();
            }
        };
    }

    impl_test_fn_for_model!(test_aerouc5, "aerouc5.fts.xml");
    impl_test_fn_for_model!(test_claroline, "claroline-pow.fts.xml");
    impl_test_fn_for_model!(test_cpterminal, "cpterminal.fts.xml");
    impl_test_fn_for_model!(test_minepump, "minepump.fts.xml");
    impl_test_fn_for_model!(test_svm, "svm.fts.xml");
}
