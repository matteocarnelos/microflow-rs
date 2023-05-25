use nalgebra::SMatrix;

pub type Buffer2D<T, const ROWS: usize, const COLS: usize> = SMatrix<T, ROWS, COLS>;
pub type Buffer4D<
    T,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANNELS: usize,
> = [SMatrix<[T; CHANNELS], ROWS, COLS>; BATCHES];
