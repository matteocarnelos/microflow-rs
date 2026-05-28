#[derive(Clone, Copy)]
pub struct Job {
    #[doc(hidden)]
    /// Type erased `worker` function
    func: unsafe fn(*mut ()),
    #[doc(hidden)]
    /// Type erased argument to the `worker` function
    arg: *mut (),
}

// Safety: Each job works on a exclusive slice of memory of a Send + Sync type
unsafe impl Send for Job {}
impl Job {
    /// Create a new job
    ///
    /// # Safety
    /// `arg` has to represent a valid argument for the unsafe `func`
    ///
    unsafe fn new(func: unsafe fn(*mut ()), arg: *mut ()) -> Self {
        Job { func, arg }
    }

    /// Runs the job and consumes it
    pub fn run(self) {
        // SAFETY: this type cannot be constructed outside of this crate
        // and the safety requirements are forwarded to the struct's construction
        unsafe { (self.func)(self.arg) };
    }
}
/// Trait for pluggable computation backends.
///
/// A backend controls how the inner computation loops of the inference ops
/// are scheduled.
///
/// `defer_job` is given a pointer-sized `arg`. The backend **must** ensure
/// all deferred closures have completed before `wait` returns and before the
/// data pointed to by `arg` is dropped or moved.
pub trait Backend {
    /// Schedule a single unit of work.
    fn defer_job(job: Job);

    /// Block until all jobs previously submitted via [`defer_job`] have completed.
    fn wait();

    /// Evaluates a function across a 2D grid, potentially in parallel.
    fn from_fn<T: Send + Copy, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        // SAFETY: T is always a quantized integer type (or array thereof) for which
        // zero bytes are a valid representation. Workers overwrite every cell before
        // `wait()` returns, so no zeroed value is ever observed.
        let mut output: [[T; ROWS]; COLS] = unsafe { core::mem::zeroed() };

        struct ParallelizationUnit<'a, F, T, const R: usize> {
            func: &'a F,
            output_col: &'a mut [T; R],
            col: usize,
        }

        /// Runs a function on a 1 x R column
        ///
        /// # Safety
        ///
        /// The caller must ensure that `arg` represents pointer to an **initialized** `Job<'_, F, T, R>`
        ///
        unsafe fn worker<F, T, const R: usize>(arg: *mut ())
        where
            F: Fn(usize, usize) -> T + Sync,
            T: Send,
        {
            // SAFETY: the safety requirements are forwarded to the caller
            let job =
                unsafe { (arg.cast::<ParallelizationUnit<'_, F, T, R>>()).as_mut_unchecked() };
            for row in 0..R {
                let value = (job.func)(job.col, row);
                job.output_col[row] = value;
            }
        }

        for (c_i, column) in output.iter_mut().enumerate() {
            let mut job = ParallelizationUnit {
                func: &func,
                output_col: column,
                col: c_i,
            };
            Self::defer_job(unsafe {
                Job::new(worker::<F, T, ROWS>, (core::ptr::from_mut(&mut job)).cast())
            });
        }
        Self::wait();

        crate::buffer::Buffer2D::from_array_storage(nalgebra::ArrayStorage(output))
    }
}

/// The default sequential backend.
///
/// Every job submitted is run immediately on the calling thread.
pub struct SequentialBackend;

impl Backend for SequentialBackend {
    /// This should never be reached because all of the work is done
    /// in `from_fn`
    #[inline(always)]
    fn defer_job(_job: Job) {
        unreachable!()
    }

    /// This should never be reached because all of the work is done
    /// in `from_fn`
    #[inline(always)]
    fn wait() {
        unreachable!()
    }

    #[inline(always)]
    fn from_fn<T: Send + Copy, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        let output: [[T; ROWS]; COLS] =
            core::array::from_fn(|column| core::array::from_fn(|row| func(row, column)));
        crate::buffer::Buffer2D::from_array_storage(nalgebra::ArrayStorage(output))
    }
}
