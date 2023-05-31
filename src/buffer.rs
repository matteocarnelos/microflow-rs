use nalgebra::SMatrix;

pub type Buffer2D<T, const ROWS: usize, const COLUMNS: usize> = SMatrix<T, ROWS, COLUMNS>;
pub type Buffer4D<
    T,
    const BATCHES: usize,
    const ROWS: usize,
    const COLUMNS: usize,
    const CHANNELS: usize,
> = [Buffer2D<[T; CHANNELS], ROWS, COLUMNS>; BATCHES];
