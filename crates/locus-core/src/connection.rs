use crate::error::CoreError;
use crate::tf::TransferFunction;

/// Series (cascade) connection: `G(s) = G1(s) * G2(s)`.
pub fn series(g1: &TransferFunction, g2: &TransferFunction) -> Result<TransferFunction, CoreError> {
    let num = g1.num() * g2.num();
    let den = g1.den() * g2.den();
    TransferFunction::new(num, den)
}

/// Parallel connection: `G(s) = G1(s) + G2(s)`.
pub fn parallel(
    g1: &TransferFunction,
    g2: &TransferFunction,
) -> Result<TransferFunction, CoreError> {
    let n1 = g1.num() * g2.den();
    let n2 = g2.num() * g1.den();
    let num = &n1 + &n2;
    let den = g1.den() * g2.den();
    TransferFunction::new(num, den)
}

/// Negative feedback: forward path `G`, feedback path `H`,
/// closed loop `T = G / (1 + G*H)`.
pub fn feedback(g: &TransferFunction, h: &TransferFunction) -> Result<TransferFunction, CoreError> {
    let num = g.num() * h.den();
    let prod = g.num() * h.num();
    let dh = g.den() * h.den();
    let den = &dh + &prod;
    TransferFunction::new(num, den)
}

/// Positive feedback: `T = G / (1 - G*H)`.
pub fn feedback_positive(
    g: &TransferFunction,
    h: &TransferFunction,
) -> Result<TransferFunction, CoreError> {
    let num = g.num() * h.den();
    let prod = g.num() * h.num();
    let dh = g.den() * h.den();
    let den = &dh - &prod;
    TransferFunction::new(num, den)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_multiplies_first_order_lags() {
        // 1/(s+1) * 1/(s+2) = 1/(s^2 + 3s + 2)
        let g1 = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 1.0]).unwrap();
        let g2 = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 2.0]).unwrap();
        let g = series(&g1, &g2).unwrap();
        assert_eq!(g.num().coeffs(), &[1.0]);
        assert_eq!(g.den().coeffs(), &[1.0, 3.0, 2.0]);
    }

    #[test]
    fn parallel_sums_first_order_lags() {
        // 1/(s+1) + 1/(s+2) = (2s + 3)/(s^2 + 3s + 2)
        let g1 = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 1.0]).unwrap();
        let g2 = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 2.0]).unwrap();
        let g = parallel(&g1, &g2).unwrap();
        assert_eq!(g.num().coeffs(), &[2.0, 3.0]);
        assert_eq!(g.den().coeffs(), &[1.0, 3.0, 2.0]);
    }

    #[test]
    fn unity_feedback_of_integrator() {
        // G = 1/s, H = 1, T = 1/(s + 1)
        let g = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 0.0]).unwrap();
        let h = TransferFunction::from_coeffs(vec![1.0], vec![1.0]).unwrap();
        let t = feedback(&g, &h).unwrap();
        assert_eq!(t.num().coeffs(), &[1.0]);
        assert_eq!(t.den().coeffs(), &[1.0, 1.0]);
    }

    #[test]
    fn unity_feedback_of_double_integrator_with_damping() {
        // G = 1/(s^2 + s), H = 1, T = 1/(s^2 + s + 1)
        let g = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 1.0, 0.0]).unwrap();
        let h = TransferFunction::from_coeffs(vec![1.0], vec![1.0]).unwrap();
        let t = feedback(&g, &h).unwrap();
        assert_eq!(t.num().coeffs(), &[1.0]);
        assert_eq!(t.den().coeffs(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn positive_feedback_unstable_loop() {
        // G = 1/(s+1), H = 1, positive feedback: T = 1/((s+1) - 1) = 1/s
        let g = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 1.0]).unwrap();
        let h = TransferFunction::from_coeffs(vec![1.0], vec![1.0]).unwrap();
        let t = feedback_positive(&g, &h).unwrap();
        assert_eq!(t.num().coeffs(), &[1.0]);
        assert_eq!(t.den().coeffs(), &[1.0, 0.0]);
    }
}
