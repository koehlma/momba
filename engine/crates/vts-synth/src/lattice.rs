//! Data structures and abstractions for partially ordered sets.

use std::hash::Hash;

/// A type whose values are partially ordered.
pub trait PartiallyOrdered: Eq {
    /// Returns `true` if and only if `self` is strictly less than `other`.
    fn is_less_than(&self, other: &Self) -> bool;

    /// Returns `true` if and only if `self` is strictly greater than `other`.
    #[inline]
    fn is_greater_than(&self, other: &Self) -> bool {
        other.is_less_than(self)
    }

    /// Returns `true` if and only if `self` is at most `other`.
    #[inline]
    fn is_at_most(&self, other: &Self) -> bool {
        self.is_less_than(other) || self == other
    }

    /// Returns `true` if and only if `self` is at least `other`.
    #[inline]
    fn is_at_least(&self, other: &Self) -> bool {
        self.is_greater_than(other) || self == other
    }

    /// Returns `true` if and only if `self` and `other` are incomparable.
    #[inline]
    fn are_incomparable(&self, other: &Self) -> bool {
        !self.is_less_than(other) && !other.is_less_than(self) && self != other
    }
}

/// A type with a least upper bound (_meet_) for any pair of values.
pub trait Meet: PartiallyOrdered {
    /// Computes the meet of `self` and `other`.
    fn meet(&self, other: &Self) -> Self;

    /// Computes the meet of `self` and `other` and stores the result in `self`.
    ///
    /// Returns `true` if `self` has changed.
    #[inline]
    fn meet_assign(&mut self, other: &Self) -> bool
    where
        Self: Sized,
    {
        let mut meet = self.meet(other);
        std::mem::swap(self, &mut meet);
        return *self != meet;
    }

    /// Computes the meet of `self` and `other`.
    #[inline]
    fn meet_move(self, other: Self) -> Self
    where
        Self: Sized,
    {
        self.meet(&other)
    }
}

/// A type with a greatest lower bound (_join_) for any pair of values.
pub trait Join: PartiallyOrdered {
    /// Computes the join of `self` and `other`.
    fn join(&self, other: &Self) -> Self;

    /// Computes the join of `self` and `other` and stores the result in `self`.
    ///
    /// Returns `true` if `self` has changed.
    #[inline]
    fn join_assign(&mut self, other: &Self) -> bool
    where
        Self: Sized,
    {
        let mut join = self.join(other);
        std::mem::swap(self, &mut join);
        return *self != join;
    }

    /// Computes the join of `self` and `other`.
    #[inline]
    fn join_move(self, other: Self) -> Self
    where
        Self: Sized,
    {
        self.join(&other)
    }
}

/// A partially ordered set.
pub trait Poset {
    /// The element type of the set implementing [`PartiallyOrdered`].
    type Element: PartiallyOrdered;
}

/// An upper-bounded partially ordered set with a unique top element.
pub trait HasTop: Poset {
    /// Returns the top element.
    fn top(&self) -> Self::Element;

    /// Checks whether the given `element` is the top element.
    fn is_top(&self, element: &Self::Element) -> bool;
}

/// A lower-bounded partially ordered set with a unique bottom element.
pub trait HasBottom: Poset {
    /// Returns the bottom element.
    fn bottom(&self) -> Self::Element;

    /// Checks whether the given `element` is the bottom element.
    fn is_bottom(&self, element: &Self::Element) -> bool;
}

/// A join-semilattice is a partially ordered set where every pair of elements has a join.
///
/// This trait is automatically implemented for every eligible [`Poset`].
pub trait JoinSemiLattice: Poset {}

impl<T: Poset> JoinSemiLattice for T where T::Element: Join {}

/// A meet-semilattice is a partially ordered set where every pair of elements has a meet.
///
/// This trait is automatically implemented for every eligible [`Poset`].
pub trait MeetSemiLattice: Poset {}

impl<T: Poset> MeetSemiLattice for T where T::Element: Meet {}

/// A lattice is a partially ordered set where every pair of elements has a join and meet.
///
/// This trait is automatically implemented for every eligible [`Poset`].
pub trait Lattice: JoinSemiLattice + MeetSemiLattice {}

impl<T: Poset> Lattice for T where T::Element: Meet + Join {}

/// A set of elements of type `T`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Set<T: Clone + Hash + Eq>(pub(crate) im::HashSet<T>);

impl<T: Clone + Hash + Eq> Set<T> {
    pub fn new() -> Self {
        Self(im::HashSet::new())
    }

    /// Inserts an element into the set.
    pub fn insert(&mut self, element: T) {
        self.0.insert(element);
    }

    /// Removes an element from the set.
    pub fn remove(&mut self, element: &T) -> Option<T> {
        self.0.remove(element)
    }

    /// An iterator over the set's elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}

impl<'s, T: Clone + Hash + Eq> IntoIterator for &'s Set<T> {
    type Item = &'s T;

    type IntoIter = im::hashset::Iter<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T: Clone + Hash + Eq> PartiallyOrdered for Set<T> {
    fn is_less_than(&self, other: &Self) -> bool {
        self.0.is_proper_subset(&other.0)
    }

    fn is_at_most(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }
}

impl<T: Clone + Hash + Eq> Meet for Set<T> {
    fn meet(&self, other: &Self) -> Self {
        Self(self.0.clone().intersection(other.0.clone()))
    }
}

impl<T: Clone + Hash + Eq> Join for Set<T> {
    fn join(&self, other: &Self) -> Self {
        Self(self.0.clone().union(other.0.clone()))
    }
}

// impl<T: Clone + Hash + Eq> HasBottom for Set<T> {
//     fn is_bottom(&self) -> bool {
//         self.0.is_empty()
//     }
// }

impl<T: Clone + Hash + Eq> Default for Set<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: Clone + Hash + Eq> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// The dual of a partially ordered type `T`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dual<T>(T);

impl<T> Dual<T> {
    pub fn new(inner: T) -> Self {
        Self(inner)
    }

    pub fn into_inner(self) -> T {
        self.0
    }

    pub fn inner(&self) -> &T {
        &self.0
    }
}

impl<T: PartiallyOrdered> PartiallyOrdered for Dual<T> {
    fn is_less_than(&self, other: &Self) -> bool {
        self.0.is_greater_than(&other.0)
    }

    fn is_greater_than(&self, other: &Self) -> bool {
        self.0.is_less_than(&other.0)
    }

    fn is_at_most(&self, other: &Self) -> bool {
        self.0.is_at_least(&other.0)
    }

    fn is_at_least(&self, other: &Self) -> bool {
        self.0.is_at_most(&other.0)
    }

    fn are_incomparable(&self, other: &Self) -> bool {
        self.0.are_incomparable(&other.0)
    }
}

impl<T: Meet> Join for Dual<T> {
    fn join(&self, other: &Self) -> Self {
        Self(self.0.meet(&other.0))
    }

    fn join_assign(&mut self, other: &Self) -> bool
    where
        Self: Sized,
    {
        self.0.meet_assign(&other.0)
    }
}

impl<T: Join> Meet for Dual<T> {
    fn meet(&self, other: &Self) -> Self {
        Self(self.0.join(&other.0))
    }

    fn meet_assign(&mut self, other: &Self) -> bool
    where
        Self: Sized,
    {
        self.0.join_assign(&other.0)
    }
}

impl<T: Poset> Poset for Dual<T> {
    type Element = Dual<T::Element>;
}

impl<T: HasBottom> HasTop for Dual<T> {
    fn top(&self) -> Self::Element {
        Dual(self.0.bottom())
    }

    fn is_top(&self, element: &Self::Element) -> bool {
        self.0.is_bottom(&element.0)
    }
}

impl<T: HasTop> HasBottom for Dual<T> {
    fn bottom(&self) -> Self::Element {
        Dual(self.0.top())
    }

    fn is_bottom(&self, element: &Self::Element) -> bool {
        self.0.is_top(&element.0)
    }
}
