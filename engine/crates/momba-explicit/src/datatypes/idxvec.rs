use std::{marker::PhantomData, ops};

pub trait Idx {
    fn as_usize(self) -> usize;

    fn from_idx(idx: usize) -> Self;
}

#[derive(Debug, Clone)]
pub struct IdxVec<I, T> {
    vec: Vec<T>,
    _phantom_idx: PhantomData<fn(&I)>,
}

impl<I, T> IdxVec<I, T> {
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
            _phantom_idx: PhantomData,
        }
    }
}

impl<I, T> From<Vec<T>> for IdxVec<I, T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            vec: value,
            _phantom_idx: PhantomData,
        }
    }
}

impl<I: Idx, T> IdxVec<I, T> {
    pub fn next_idx(&self) -> I {
        I::from_idx(self.vec.len())
    }

    pub fn indexed_iter(&self) -> impl '_ + Iterator<Item = (I, &T)> {
        self.vec
            .iter()
            .enumerate()
            .map(|(idx, item)| (I::from_idx(idx), item))
    }
}

impl<I, T> Default for IdxVec<I, T> {
    fn default() -> Self {
        Self {
            vec: Default::default(),
            _phantom_idx: Default::default(),
        }
    }
}

impl<I, T> ops::Deref for IdxVec<I, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<I, T> ops::DerefMut for IdxVec<I, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl<I, T> ops::Index<usize> for IdxVec<I, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec[index]
    }
}

impl<I: Idx, T> ops::Index<I> for IdxVec<I, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.vec[index.as_usize()]
    }
}

impl<I: Idx, T> ops::Index<ops::Range<I>> for IdxVec<I, T> {
    type Output = [T];

    fn index(&self, index: ops::Range<I>) -> &Self::Output {
        &self.vec[index.start.as_usize()..index.end.as_usize()]
    }
}

macro_rules! new_idx_type {
    ($(#[$($meta:meta)*])* $vis:vis $name:ident ( $type:ty )) => {
        $(#[$($meta)*])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        $vis struct $name($type);

        impl $crate::datatypes::idxvec::Idx for $name {
            fn as_usize(self) -> usize {
                self.0 as usize
            }

            fn from_idx(idx: usize) -> Self {
                Self(idx as $type)
            }
        }

        impl From<usize> for $name {
            fn from(idx: usize) -> Self {
                match <$type>::try_from(idx) {
                    Ok(idx) => Self(idx),
                    Err(_) => {
                        panic!("Index overflow.")
                    }
                }
            }
        }
    };
}

pub(crate) use new_idx_type;
