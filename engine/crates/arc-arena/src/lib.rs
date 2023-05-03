//! A fast allocation arena with atomically reference counted pages.

use std::{
    alloc::Layout,
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::{drop_in_place, NonNull},
    sync::atomic,
};

/// The default size of pages (1 MiB).
pub const DEFAULT_PAGE_SIZE: usize = 1 * 1024 * 1024;

struct PageHeader {
    /// The reference counter for the page.
    count: atomic::AtomicUsize,
}

struct PageLayout {
    header_layout: Layout,
    page_layout: Layout,
    capacity: usize,
}

struct Page<const PAGE_SIZE: usize>(NonNull<PageHeader>);

impl<const PAGE_SIZE: usize> Page<PAGE_SIZE> {
    const fn layout() -> PageLayout {
        if !PAGE_SIZE.is_power_of_two() {
            panic!("Page size must be a power of two.");
        }
        let header_layout = Layout::new::<PageHeader>();
        if header_layout.size() > PAGE_SIZE {
            panic!("Page size is too small.");
        }
        let capacity = PAGE_SIZE - header_layout.size();
        let Ok(page_layout) = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE)  else {
            panic!("Page size is invalid.");
        };
        PageLayout {
            header_layout,
            page_layout,
            capacity,
        }
    }

    fn new() -> Self {
        let page_layout = Self::layout().page_layout;
        let Some(ptr) = NonNull::new(unsafe { std::alloc::alloc(page_layout) as *mut PageHeader }) else {
            std::alloc::handle_alloc_error(page_layout);
        };
        Self(ptr)
    }

    fn data_ptr(&self) -> *mut u8 {
        unsafe { (self.0.as_ptr() as *mut u8).offset(Self::layout().header_layout.size() as isize) }
    }

    fn header(&self) -> &PageHeader {
        unsafe { &*self.0.as_ptr() }
    }
}

type InvalidPage = Page<{ 2 * 1024 * 1024 }>;

const _: () = {
    let _ = InvalidPage::layout();
};

struct ArcPage {
    size: usize,
    counter: atomic::AtomicUsize,
    data: UnsafeCell<[MaybeUninit<u8>; 0]>,
}

impl ArcPage {
    pub fn data(&self) -> *const [MaybeUninit<u8>] {
        unsafe { std::slice::from_raw_parts((*self.data.get()).as_ptr(), self.size) }
    }
}

pub struct ArcPageAllocation<const PAGE_SIZE: usize>;

pub trait Free {
    unsafe fn free<T>(&self, ptr: *mut T);
}

impl<const PAGE_SIZE: usize> Free for ArcPageAllocation<PAGE_SIZE> {
    unsafe fn free<T>(&self, ptr: *mut T) {
        todo!()
    }
}

pub struct Box<T, A: Free> {
    ptr: *mut T,
    free: A,
}

impl<T, P: Free> Drop for Box<T, P> {
    fn drop(&mut self) {
        unsafe {
            drop_in_place(self.ptr);
            self.free.free(self.ptr);
        }
    }
}

impl<T, P: Free> Box<T, P> {
    pub fn deref(&self) -> &T {
        unsafe { &*self.ptr }
    }

    pub fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr }
    }
}

impl<T, P: Free> std::ops::Deref for Box<T, P> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.deref()
    }
}

impl<T, P: Free> std::ops::DerefMut for Box<T, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.deref_mut()
    }
}

unsafe impl<T: Send, P: Send + Free> Send for Box<T, P> {}

unsafe impl<T: Sync, P: Sync + Free> Sync for Box<T, P> {}
