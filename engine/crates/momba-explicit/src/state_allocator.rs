//!
//!
//! ```plain
//! Start          Head            Length             Capacity
//! |               |                |                   |
//! v               v                v                   v
//! +---------------+----------------+-------------------+
//! | Read-Only ... | Read-Write ... | Uninitialized ... |
//! +---------------+----------------+-------------------+
//! ```

use std::{
    alloc,
    marker::PhantomData,
    mem, ops,
    ptr::{addr_of_mut, drop_in_place, NonNull},
    sync::atomic,
};

use parking_lot::RwLock;

#[derive(Debug)]
struct PageAllocator<T> {
    /// The page memory layout.
    layout: alloc::Layout,
    /// The capacity of a page.
    capacity: usize,
    _phantom_type: PhantomData<fn(&T)>,
}

impl<T> PageAllocator<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        let size = mem::size_of::<Page<T>>() + mem::size_of::<T>() * capacity;
        todo!()
    }

    pub fn with_size(size: usize) -> Self {
        let space = size - mem::size_of::<Page<T>>();
        let capacity = if mem::size_of::<T>() == 0 {
            usize::MAX
        } else {
            space / mem::size_of::<T>()
        };
        let align = mem::align_of::<Page<T>>();
        Self {
            layout: alloc::Layout::from_size_align(size, align).unwrap(),
            capacity,
            _phantom_type: PhantomData,
        }
    }

    pub fn alloc(&self) -> NonNull<Page<T>> {
        // SAFETY: The layout has a non-zero size.
        let page = unsafe { alloc::alloc(self.layout) } as *mut Page<T>;
        if let Some(this) = NonNull::new(page) {
            // SAFETY: The pointer is valid and properly aligned.
            unsafe {
                this.as_ptr().write(Page {
                    idx: u32::MAX,
                    capacity: self.capacity,
                    head: atomic::AtomicUsize::new(0),
                    length: 0,
                    items: [],
                });
            }
            this
        } else {
            alloc::handle_alloc_error(self.layout)
        }
    }

    pub unsafe fn free(&self, page: NonNull<Page<T>>) {
        Page::drop_items(page);
        alloc::dealloc(page.as_ptr() as *mut u8, self.layout);
    }
}

#[repr(C)]
struct Page<T> {
    idx: u32,
    capacity: usize,
    head: atomic::AtomicUsize,
    length: usize,
    items: [T; 0],
}

impl<T> Page<T> {
    unsafe fn set_index(this: NonNull<Self>, idx: u32) {
        (*this.as_ptr()).idx = idx;
    }

    unsafe fn item_ptr_mut(this: NonNull<Self>, idx: usize) -> *mut T {
        let this = this.as_ptr();
        debug_assert!(idx < (*this).capacity, "Index out of range.");
        (addr_of_mut!((*this).items) as *mut T).offset(idx as isize)
    }

    unsafe fn drop_items(this: NonNull<Self>) {
        for idx in 0..this.as_ref().length {
            drop_in_place(Self::item_ptr_mut(this, idx));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key {
    page: u32,
    item: u32,
}

pub struct PageRefMut<'store, T> {
    page: NonNull<Page<T>>,
    _phantom_store: PhantomData<&'store StateStore<T>>,
}

impl<'store, T> PageRefMut<'store, T> {
    pub fn insert(&mut self, value: T) -> Key {
        let page = self.page.as_ptr();
        unsafe {
            let idx = (*page).length;
            Page::item_ptr_mut(self.page, idx).write(value);
            (*page).length += 1;
            Key {
                page: (*page).idx,
                item: idx as u32,
            }
        }
    }

    pub fn commit(&mut self) {
        let page = self.page.as_ptr();
        unsafe {
            (*page)
                .head
                .store((*page).length, atomic::Ordering::Release);
        }
    }
}

pub struct StateStore<T> {
    allocator: PageAllocator<T>,
    pages: RwLock<Vec<NonNull<Page<T>>>>,
}

const PAGE_SIZE: usize = 20 * 1024 * 1024; // 20 MiB page

impl<T> StateStore<T> {
    pub fn new() -> Self {
        Self {
            pages: RwLock::default(),
            allocator: PageAllocator::with_size(PAGE_SIZE),
        }
    }

    pub fn create_page(&self) -> PageRefMut<'_, T> {
        let page = self.allocator.alloc();
        let mut pages = self.pages.write();
        unsafe {
            Page::set_index(page, pages.len() as u32);
        }
        pages.push(page);
        PageRefMut {
            page,
            _phantom_store: PhantomData,
        }
    }

    pub fn get(&self, idx: Key) -> Option<&T> {
        let page = self.pages.read()[idx.page as usize];
        let head = unsafe { (*page.as_ptr()).head.load(atomic::Ordering::Acquire) };
        if idx.item < head as u32 {
            Some(unsafe { &*Page::item_ptr_mut(page, idx.item as usize) })
        } else {
            None
        }
    }
}

impl<T> Drop for StateStore<T> {
    fn drop(&mut self) {
        for page in self.pages.get_mut() {
            unsafe { self.allocator.free(*page) }
        }
    }
}

impl<T> ops::Index<Key> for StateStore<T> {
    type Output = T;

    fn index(&self, key: Key) -> &Self::Output {
        self.get(key).unwrap()
    }
}
