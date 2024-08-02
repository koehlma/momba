//! Parser for propositional logic formulae.

use std::{
    fmt::{self, Write},
    ops::Range,
};

use ariadne::{Color, Fmt, Label, Report, ReportKind, Source};
use chumsky::{error::SimpleReason, prelude::*, Stream};

use super::Formula;

type Span = Range<usize>;

type Spanned<T> = (T, Span);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Token {
    Identifier(String),

    ParenthesisOpen,
    ParenthesisClose,

    OperatorNot,
    OperatorAnd,
    OperatorOr,
    OperatorImplies,
    OperatorIff,
    OperatorXor,

    ConstantTrue,
    ConstantFalse,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Identifier(identifier) => f.write_str(identifier),
            Token::ParenthesisOpen => f.write_char('('),
            Token::ParenthesisClose => f.write_char(')'),
            Token::OperatorNot => f.write_char('!'),
            Token::OperatorAnd => f.write_str("and"),
            Token::OperatorOr => f.write_str("or"),
            Token::OperatorImplies => f.write_str("==>"),
            Token::OperatorIff => f.write_str("<=>"),
            Token::OperatorXor => f.write_str("xor"),
            Token::ConstantTrue => f.write_str("True"),
            Token::ConstantFalse => f.write_str("False"),
        }
    }
}

/// Lexer for propositional logic formulae.
fn lexer() -> impl Parser<char, Vec<Spanned<Token>>, Error = Simple<char>> {
    fn lexer_operator_not() -> impl Parser<char, Token, Error = Simple<char>> {
        just('!').or(just('~')).or(just('¬')).to(Token::OperatorNot)
    }

    fn lexer_operator_and() -> impl Parser<char, Token, Error = Simple<char>> {
        just("&&")
            .or(just("&"))
            .or(just("AND"))
            .or(just("and"))
            .or(just("∧"))
            .to(Token::OperatorAnd)
    }

    fn lexer_operator_or() -> impl Parser<char, Token, Error = Simple<char>> {
        just("||")
            .or(just("|"))
            .or(just("OR"))
            .or(just("or"))
            .or(just("v"))
            .or(just("∨"))
            .to(Token::OperatorOr)
    }

    fn lexer_operator_xor() -> impl Parser<char, Token, Error = Simple<char>> {
        just("XOR").or(just("xor")).to(Token::OperatorXor)
    }

    fn lexer_operator_implies() -> impl Parser<char, Token, Error = Simple<char>> {
        just("==>")
            .or(just("=>"))
            .or(just("->"))
            .or(just("⇒"))
            .or(just("→"))
            .to(Token::OperatorImplies)
    }

    fn lexer_operator_iff() -> impl Parser<char, Token, Error = Simple<char>> {
        just("<=>")
            .or(just("<->"))
            .or(just("⇔"))
            .or(just("↔"))
            .to(Token::OperatorIff)
    }

    fn lexer_constant_true() -> impl Parser<char, Token, Error = Simple<char>> {
        just("1").or(just("⊤")).to(Token::ConstantTrue)
    }

    fn lexer_constant_false() -> impl Parser<char, Token, Error = Simple<char>> {
        just("0").or(just("⊥")).to(Token::ConstantFalse)
    }

    let token = text::ident()
        .map(|identifier: String| match identifier.as_str() {
            "True" | "true" => Token::ConstantTrue,
            "False" | "false" => Token::ConstantFalse,
            _ => Token::Identifier(identifier),
        })
        .or(just('(').to(Token::ParenthesisOpen))
        .or(just(')').to(Token::ParenthesisClose))
        .or(lexer_operator_not())
        .or(lexer_operator_and())
        .or(lexer_operator_or())
        .or(lexer_operator_implies())
        .or(lexer_operator_iff())
        .or(lexer_operator_xor())
        .or(lexer_constant_true())
        .or(lexer_constant_false());

    token
        .map_with_span(|token, span| (token, span))
        .padded()
        .repeated()
        .then_ignore(end())
}

/// Parser for propositional logic formulae.
fn parser() -> impl Parser<Token, Formula<String>, Error = Simple<Token>> {
    recursive(|formula| {
        let primary = formula
            .delimited_by(just(Token::ParenthesisOpen), just(Token::ParenthesisClose))
            .or(select! {
                Token::Identifier(identifier) => Formula::Atom(identifier),
                Token::ConstantTrue => Formula::True,
                Token::ConstantFalse => Formula::False,
            });

        let negation = just(Token::OperatorNot)
            .repeated()
            .then(primary)
            .foldr(|_, operand| operand.not());

        let xor = negation
            .clone()
            .then(just(Token::OperatorXor).ignore_then(negation).repeated())
            .foldl(|left, right| left.xor(right));

        let conjunction = xor
            .clone()
            .then(just(Token::OperatorAnd).ignore_then(xor).repeated())
            .foldl(|left, right| left.and(right));

        let disjunction = conjunction
            .clone()
            .then(just(Token::OperatorOr).ignore_then(conjunction).repeated())
            .foldl(|left, right| left.or(right));

        let implication = recursive(|implication| {
            disjunction
                .then(
                    just(Token::OperatorImplies)
                        .ignore_then(implication)
                        .or_not(),
                )
                .map(|(left, right)| {
                    if let Some(right) = right {
                        left.not().or(right)
                    } else {
                        left
                    }
                })
        });

        let equivalence = recursive(|equivalence| {
            implication
                .then(just(Token::OperatorIff).ignore_then(equivalence).or_not())
                .map(|(left, right): (_, Option<Formula<String>>)| {
                    if let Some(right) = right {
                        left.clone()
                            .and(right.clone())
                            .or(left.not().and(right.not()))
                    } else {
                        left
                    }
                })
        });

        equivalence
    })
    .then_ignore(end())
}

#[derive(Debug)]
pub struct Error(Simple<String>);

impl Error {
    pub fn eprint(&self, formula: &str) {
        let report = Report::build(ReportKind::Error, (), self.0.span().start);

        match self.0.reason() {
            SimpleReason::Unclosed { span, delimiter } => report
                .with_message(format!(
                    "Unclosed delimiter {}",
                    delimiter.fg(Color::Yellow)
                ))
                .with_label(
                    Label::new(span.clone())
                        .with_message(format!(
                            "Unclosed delimiter {}",
                            delimiter.fg(Color::Yellow)
                        ))
                        .with_color(Color::Yellow),
                )
                .with_label(
                    Label::new(self.0.span())
                        .with_message(format!(
                            "Must be closed before this {}",
                            self.0
                                .found()
                                .unwrap_or(&"end of file".to_string())
                                .fg(Color::Red)
                        ))
                        .with_color(Color::Red),
                ),
            SimpleReason::Unexpected => report
                .with_message(format!(
                    "{}, expected {}",
                    if self.0.found().is_some() {
                        "Unexpected token in input"
                    } else {
                        "Unexpected end of input"
                    },
                    if self.0.expected().len() == 0 {
                        "something else".to_string()
                    } else {
                        self.0
                            .expected()
                            .map(|expected| match expected {
                                Some(expected) => expected.to_string(),
                                None => "end of input".to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ))
                .with_label(
                    Label::new(self.0.span())
                        .with_message(format!(
                            "Unexpected token {}",
                            self.0
                                .found()
                                .unwrap_or(&"end of file".to_string())
                                .fg(Color::Red)
                        ))
                        .with_color(Color::Red),
                ),
            chumsky::error::SimpleReason::Custom(msg) => report.with_message(msg).with_label(
                Label::new(self.0.span())
                    .with_message(format!("{}", msg.fg(Color::Red)))
                    .with_color(Color::Red),
            ),
        }
        .finish()
        .eprint(Source::from(formula))
        .unwrap();
    }
}

pub fn parse(formula: &str) -> (Option<Formula<String>>, Vec<Error>) {
    let (tokens, lexer_errors) = lexer().parse_recovery(formula);

    let mut errors = lexer_errors
        .into_iter()
        .map(|error| Error(error.map(|c| c.to_string())))
        .collect::<Vec<_>>();

    let result = tokens.and_then(|tokens| {
        let len = formula.chars().count();

        let (formula, parser_errors) =
            parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter()));

        errors.extend(
            parser_errors
                .into_iter()
                .map(|error| Error(error.map(|t| t.to_string()))),
        );

        formula
    });

    (result, errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_parser() {
        assert_eq!(
            parse("!a && b || c").0.unwrap(),
            Formula::Atom("a".to_owned())
                .not()
                .and(Formula::Atom("b".to_owned()))
                .or(Formula::Atom("c".to_owned()))
        );

        assert_eq!(
            parse("!a && (b || c)").0.unwrap(),
            Formula::Atom("a".to_owned())
                .not()
                .and(Formula::Atom("b".to_owned()).or(Formula::Atom("c".to_owned())))
        );
    }
}
