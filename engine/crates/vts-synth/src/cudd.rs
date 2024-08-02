//! A safe abstraction for BDDs provided by CUDD.

use std::{hash::Hash, marker::PhantomData, ptr::NonNull};

use rand::Rng;

use crate::lattice::{HasBottom, HasTop, Join, Meet, PartiallyOrdered, Poset};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trinary {
    False = 0,
    True = 1,
    Any = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReorderingType {
    Same,
    None,
    Sift,
}

impl ReorderingType {
    fn raw(&self) -> cudd_sys::cudd::Cudd_ReorderingType {
        match self {
            ReorderingType::Same => cudd_sys::cudd::Cudd_ReorderingType::CUDD_REORDER_SAME,
            ReorderingType::None => cudd_sys::cudd::Cudd_ReorderingType::CUDD_REORDER_NONE,
            ReorderingType::Sift => cudd_sys::cudd::Cudd_ReorderingType::CUDD_REORDER_SIFT,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BddManager {
    ptr: NonNull<cudd_sys::DdManager>,
}

impl BddManager {
    pub fn new() -> Self {
        Self::with_vars(0)
    }

    pub fn with_vars(num_vars: u32) -> Self {
        let manager = NonNull::new(unsafe {
            cudd_sys::cudd::Cudd_Init(
                num_vars,
                0,
                cudd_sys::cudd::CUDD_UNIQUE_SLOTS,
                cudd_sys::cudd::CUDD_CACHE_SLOTS,
                0,
            )
        })
        .expect("`Cudd_Init` returned a null pointer. Out of memory?");
        Self { ptr: manager }
    }

    pub fn one(&self) -> BddNode {
        BddNode::new(self.ptr, unsafe {
            cudd_sys::cudd::Cudd_ReadOne(self.ptr.as_ptr())
        })
    }

    pub fn zero(&self) -> BddNode {
        BddNode::new(self.ptr, unsafe {
            cudd_sys::cudd::Cudd_ReadLogicZero(self.ptr.as_ptr())
        })
    }

    pub fn var(&self, index: i32) -> BddNode {
        BddNode::new(self.ptr, unsafe {
            cudd_sys::cudd::Cudd_bddIthVar(self.ptr.as_ptr(), index)
        })
    }

    pub fn new_var(&self) -> BddNode {
        BddNode::new(self.ptr, unsafe {
            cudd_sys::cudd::Cudd_bddNewVar(self.ptr.as_ptr())
        })
    }

    pub fn reorder_now(&mut self, typ: ReorderingType) {
        unsafe { cudd_sys::cudd::Cudd_ReduceHeap(self.ptr.as_mut(), typ.raw(), 1000) };
    }

    pub fn enable_reordering(&mut self, typ: ReorderingType) {
        unsafe { cudd_sys::cudd::Cudd_AutodynEnable(self.ptr.as_mut(), typ.raw()) };
    }
}

impl Default for BddManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BddManager {
    fn drop(&mut self) {
        unsafe {
            cudd_sys::cudd::Cudd_Quit(self.ptr.as_ptr());
        }
    }
}

#[derive(Debug, PartialEq, Eq, Ord, Hash)]
pub struct BddNode<'m> {
    // We store a direct pointer to the `DdManager` instead of a reference to
    // `CuddManager` to avoid an additional indirection.
    mgr: NonNull<cudd_sys::DdManager>,
    node: NonNull<cudd_sys::DdNode>,
    _phantom_lifetime: PhantomData<&'m BddManager>,
}

impl<'m> BddNode<'m> {
    fn new(mgr: NonNull<cudd_sys::DdManager>, node: *mut cudd_sys::DdNode) -> Self {
        let node = NonNull::new(node).expect("Node pointer must not be NULL.");
        unsafe {
            cudd_sys::cudd::Cudd_Ref(node.as_ptr());
        }
        BddNode {
            mgr,
            node,
            _phantom_lifetime: PhantomData,
        }
    }

    pub fn index(&self) -> u32 {
        unsafe { cudd_sys::cudd::Cudd_NodeReadIndex(self.node.as_ptr()) }
    }

    pub fn is_constant(&self) -> bool {
        unsafe { cudd_sys::cudd::Cudd_IsConstant(self.node.as_ptr()) != 0 }
    }

    pub fn is_zero(&self) -> bool {
        self.node.as_ptr() == unsafe { cudd_sys::cudd::Cudd_ReadLogicZero(self.mgr.as_ptr()) }
    }

    pub fn is_one(&self) -> bool {
        self.node.as_ptr() == unsafe { cudd_sys::cudd::Cudd_ReadOne(self.mgr.as_ptr()) }
    }

    pub fn and(&self, other: &Self) -> Self {
        assert_eq!(
            self.mgr, other.mgr,
            "Manager of both nodes must be the same."
        );
        Self::new(self.mgr, unsafe {
            cudd_sys::cudd::Cudd_bddAnd(self.mgr.as_ptr(), self.node.as_ptr(), other.node.as_ptr())
        })
    }

    pub fn or(&self, other: &Self) -> Self {
        assert_eq!(
            self.mgr, other.mgr,
            "Manager of both nodes must be the same."
        );
        Self::new(self.mgr, unsafe {
            cudd_sys::cudd::Cudd_bddOr(self.mgr.as_ptr(), self.node.as_ptr(), other.node.as_ptr())
        })
    }

    pub fn xor(&self, other: &Self) -> Self {
        assert_eq!(
            self.mgr, other.mgr,
            "Manager of both nodes must be the same."
        );
        Self::new(self.mgr, unsafe {
            cudd_sys::cudd::Cudd_bddXor(self.mgr.as_ptr(), self.node.as_ptr(), other.node.as_ptr())
        })
    }

    pub fn complement(&self) -> Self {
        Self::new(self.mgr, unsafe {
            cudd_sys::cudd::Cudd_Complement(self.node.as_ptr())
        })
    }

    pub fn implies(&self, other: &Self) -> Self {
        self.complement().or(other)
    }

    pub fn cubes(&self) -> Vec<Vec<Trinary>> {
        let mut cubes = Vec::new();
        let num_vars = unsafe { cudd_sys::cudd::Cudd_ReadSize(self.mgr.as_ptr()) };
        unsafe {
            cudd_sys::cudd::Cudd_ForeachCube(self.mgr.as_ptr(), self.node.as_ptr(), |cube, _| {
                let mut cube_vec = Vec::new();
                for offset in 0..num_vars {
                    match *cube.offset(offset as isize) {
                        0 => cube_vec.push(Trinary::False),
                        1 => cube_vec.push(Trinary::True),
                        2 => cube_vec.push(Trinary::Any),
                        x => panic!("Invalid value `{x}` for variable in cube."),
                    }
                }
                cubes.push(cube_vec);
            })
        };
        cubes
    }

    pub fn count_solutions(&self) -> f64 {
        let num_vars = unsafe { cudd_sys::cudd::Cudd_ReadSize(self.mgr.as_ptr()) };
        unsafe {
            cudd_sys::cudd::Cudd_CountMinterm(self.mgr.as_ptr(), self.node.as_ptr(), num_vars)
        }
    }

    pub fn sample_solution(&self) -> (BddNode<'m>, Vec<bool>) {
        let mgr = self.mgr.as_ptr();
        let num_vars = unsafe { cudd_sys::cudd::Cudd_ReadSize(mgr) };
        let mut rng = rand::thread_rng();
        let mut current = self.clone();
        let solution = (0..num_vars)
            .map(|var| {
                let var = Self::new(self.mgr, unsafe {
                    cudd_sys::cudd::Cudd_bddIthVar(mgr, var)
                });
                let if_true = current.and(&var);
                let if_false = current.and(&var.complement());
                let true_count = if_true.count_solutions();
                let false_count = if_false.count_solutions();
                if rng.gen_bool(true_count / (true_count + false_count)) {
                    current = if_true;
                    true
                } else {
                    current = if_false;
                    false
                }
            })
            .collect();
        debug_assert_eq!(current.count_solutions(), 1.0);
        (current, solution)
    }
}

impl<'m> PartialOrd for BddNode<'m> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.mgr != other.mgr {
            return None;
        }
        self.index().partial_cmp(&other.index())
    }
}

impl<'m> Clone for BddNode<'m> {
    fn clone(&self) -> Self {
        unsafe {
            cudd_sys::cudd::Cudd_Ref(self.node.as_ptr());
        }
        Self {
            mgr: self.mgr,
            node: self.node,
            _phantom_lifetime: PhantomData,
        }
    }
}

impl<'m> Drop for BddNode<'m> {
    fn drop(&mut self) {
        unsafe {
            cudd_sys::cudd::Cudd_RecursiveDeref(self.mgr.as_ptr(), self.node.as_ptr());
        }
    }
}

impl<'m> PartiallyOrdered for BddNode<'m> {
    fn is_less_than(&self, other: &Self) -> bool {
        self != other && self.and(other) == *self
    }
}

// impl<'m> HasTop for BddNode<'m> {
//     fn is_top(&self) -> bool {
//         self.is_one()
//     }
// }

// impl<'m> HasBottom for BddNode<'m> {
//     fn is_bottom(&self) -> bool {
//         self.is_zero()
//     }
// }

impl<'m> Join for BddNode<'m> {
    fn join(&self, other: &Self) -> Self {
        self.or(other)
    }
}

impl<'m> Meet for BddNode<'m> {
    fn meet(&self, other: &Self) -> Self {
        self.and(other)
    }
}

impl<'m> Poset for &'m BddManager {
    type Element = BddNode<'m>;
}

impl<'m> HasBottom for &'m BddManager {
    fn bottom(&self) -> Self::Element {
        self.zero()
    }

    fn is_bottom(&self, element: &Self::Element) -> bool {
        element.is_zero()
    }
}

impl<'m> HasTop for &'m BddManager {
    fn top(&self) -> Self::Element {
        self.one()
    }

    fn is_top(&self, element: &Self::Element) -> bool {
        element.is_one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test() {
        let manager = BddManager::new();

        let zero = manager.zero();
        let one = manager.one();

        assert_eq!(one.and(&one), manager.one());
        assert_eq!(zero.and(&zero), manager.zero());
        assert_eq!(zero.and(&one), manager.zero());

        let x = manager.new_var();
        assert_eq!(x.index(), 0);
        assert_eq!(manager.var(0), x);

        let y = manager.new_var();

        assert_eq!(one.and(&x), x);
        assert_eq!(zero.and(&y), zero);
    }
}
