use crate::error::CoreError;
use crate::polynomial::Polynomial;
use num_complex::Complex64;

/// Single-input single-output linear time-invariant transfer function in
/// the Laplace domain: `G(s) = num(s) / den(s)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferFunction {
    num: Polynomial,
    den: Polynomial,
}

impl TransferFunction {
    pub fn new(num: Polynomial, den: Polynomial) -> Result<Self, CoreError> {
        if den.is_zero() {
            return Err(CoreError::ZeroDenominator);
        }
        if !num.is_zero() && num.degree() > den.degree() {
            return Err(CoreError::Improper {
                num_deg: num.degree(),
                den_deg: den.degree(),
            });
        }
        Ok(Self { num, den })
    }

    /// Convenience: build from raw coefficient vectors in descending order.
    pub fn from_coeffs(num: Vec<f64>, den: Vec<f64>) -> Result<Self, CoreError> {
        Self::new(Polynomial::new(num)?, Polynomial::new(den)?)
    }

    pub fn num(&self) -> &Polynomial {
        &self.num
    }
    pub fn den(&self) -> &Polynomial {
        &self.den
    }

    /// System order (degree of denominator).
    pub fn order(&self) -> usize {
        self.den.degree()
    }

    pub fn is_proper(&self) -> bool {
        self.num.is_zero() || self.num.degree() <= self.den.degree()
    }

    pub fn is_strictly_proper(&self) -> bool {
        self.num.is_zero() || self.num.degree() < self.den.degree()
    }

    /// Evaluate `G(s)` at a complex frequency.
    pub fn eval(&self, s: Complex64) -> Complex64 {
        self.num.eval_complex(s) / self.den.eval_complex(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn second_order_constructs() {
        let g = TransferFunction::from_coeffs(vec![1.0], vec![1.0, 2.0, 1.0]).unwrap();
        assert_eq!(g.order(), 2);
        assert!(g.is_strictly_proper());
    }

    #[test]
    fn improper_rejected() {
        assert!(matches!(
            TransferFunction::from_coeffs(vec![1.0, 1.0, 1.0], vec![1.0, 1.0]),
            Err(CoreError::Improper { .. })
        ));
    }

    #[test]
    fn zero_denominator_rejected() {
        let r = TransferFunction::new(Polynomial::new(vec![1.0]).unwrap(), Polynomial::zero());
        assert_eq!(r.unwrap_err(), CoreError::ZeroDenominator);
    }

    #[test]
    fn proper_but_not_strictly_proper() {
        // (s+1)/(s+2): equal degrees
        let g = TransferFunction::from_coeffs(vec![1.0, 1.0], vec![1.0, 2.0]).unwrap();
        assert!(g.is_proper());
        assert!(!g.is_strictly_proper());
    }

    #[test]
    fn dc_gain_of_first_order_lag() {
        // G(s) = 5/(s+10) -> G(0) = 0.5
        let g = TransferFunction::from_coeffs(vec![5.0], vec![1.0, 10.0]).unwrap();
        let v = g.eval(Complex64::new(0.0, 0.0));
        assert_relative_eq!(v.re, 0.5);
        assert_relative_eq!(v.im, 0.0);
    }
}
