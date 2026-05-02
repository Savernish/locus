use crate::error::CoreError;
use num_complex::Complex64;
use std::ops::{Add, Mul, Neg, Sub};

/// Polynomial in a single variable with f64 coefficients.
///
/// Coefficients are stored in DESCENDING power order:
/// `[a_n, a_{n-1}, ..., a_1, a_0]` represents
/// `a_n * s^n + a_{n-1} * s^{n-1} + ... + a_1 * s + a_0`.
///
/// This matches MATLAB's `tf()` convention.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    coeffs: Vec<f64>,
}

impl Polynomial {
    /// Construct from a coefficient vector in descending power order.
    /// Leading zeros are trimmed. The zero polynomial is normalized to `[0.0]`.
    pub fn new(coeffs: Vec<f64>) -> Result<Self, CoreError> {
        if coeffs.is_empty() {
            return Err(CoreError::EmptyCoefficients);
        }
        Ok(Self::trim(coeffs))
    }

    pub fn from_slice(coeffs: &[f64]) -> Result<Self, CoreError> {
        Self::new(coeffs.to_vec())
    }

    pub fn zero() -> Self {
        Self { coeffs: vec![0.0] }
    }

    pub fn one() -> Self {
        Self { coeffs: vec![1.0] }
    }

    /// Polynomial degree. The zero polynomial reports degree 0; check
    /// `is_zero()` first when the distinction matters.
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0.0)
    }

    /// Evaluate at real `x` via Horner's method.
    pub fn eval(&self, x: f64) -> f64 {
        let mut acc = 0.0;
        for &c in &self.coeffs {
            acc = acc * x + c;
        }
        acc
    }

    /// Evaluate at complex `s` via Horner's method.
    pub fn eval_complex(&self, s: Complex64) -> Complex64 {
        let mut acc = Complex64::new(0.0, 0.0);
        for &c in &self.coeffs {
            acc = acc * s + c;
        }
        acc
    }

    /// Derivative dP/ds.
    pub fn derivative(&self) -> Self {
        let n = self.degree();
        if n == 0 {
            return Self::zero();
        }
        let coeffs: Vec<f64> = self.coeffs[..self.coeffs.len() - 1]
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (n - i) as f64)
            .collect();
        Self::trim(coeffs)
    }

    fn trim(mut coeffs: Vec<f64>) -> Self {
        match coeffs.iter().position(|&c| c != 0.0) {
            Some(i) if i > 0 => {
                coeffs.drain(..i);
            }
            None => {
                coeffs = vec![0.0];
            }
            _ => {}
        }
        Self { coeffs }
    }
}

impl Add for &Polynomial {
    type Output = Polynomial;
    fn add(self, other: &Polynomial) -> Polynomial {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coeffs.iter().rev().enumerate() {
            result[n - 1 - i] += c;
        }
        for (i, &c) in other.coeffs.iter().rev().enumerate() {
            result[n - 1 - i] += c;
        }
        Polynomial::trim(result)
    }
}

impl Add for Polynomial {
    type Output = Polynomial;
    fn add(self, other: Polynomial) -> Polynomial {
        &self + &other
    }
}

impl Sub for &Polynomial {
    type Output = Polynomial;
    fn sub(self, other: &Polynomial) -> Polynomial {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coeffs.iter().rev().enumerate() {
            result[n - 1 - i] += c;
        }
        for (i, &c) in other.coeffs.iter().rev().enumerate() {
            result[n - 1 - i] -= c;
        }
        Polynomial::trim(result)
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;
    fn sub(self, other: Polynomial) -> Polynomial {
        &self - &other
    }
}

impl Mul for &Polynomial {
    type Output = Polynomial;
    fn mul(self, other: &Polynomial) -> Polynomial {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        Polynomial::trim(result)
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;
    fn mul(self, other: Polynomial) -> Polynomial {
        &self * &other
    }
}

impl Mul<f64> for &Polynomial {
    type Output = Polynomial;
    fn mul(self, scalar: f64) -> Polynomial {
        let coeffs = self.coeffs.iter().map(|&c| c * scalar).collect();
        Polynomial::trim(coeffs)
    }
}

impl Mul<f64> for Polynomial {
    type Output = Polynomial;
    fn mul(self, scalar: f64) -> Polynomial {
        &self * scalar
    }
}

impl Neg for Polynomial {
    type Output = Polynomial;
    fn neg(self) -> Polynomial {
        let coeffs = self.coeffs.iter().map(|&c| -c).collect();
        Polynomial { coeffs }
    }
}

/// Polynomial long division. Returns `(quotient, remainder)` such that
/// `num = quotient * den + remainder` with `deg(remainder) < deg(den)`.
pub fn poly_div(num: &Polynomial, den: &Polynomial) -> Result<(Polynomial, Polynomial), CoreError> {
    if den.is_zero() {
        return Err(CoreError::DivisionByZero);
    }
    if num.is_zero() {
        return Ok((Polynomial::zero(), Polynomial::zero()));
    }
    if num.degree() < den.degree() {
        return Ok((Polynomial::zero(), num.clone()));
    }

    let mut remainder = num.coeffs.clone();
    let den_lead = den.coeffs[0];
    let den_deg = den.degree();
    let mut quotient = vec![0.0; num.degree() - den_deg + 1];

    while remainder.len() >= den.coeffs.len() && !remainder.iter().all(|&c| c == 0.0) {
        let rem_deg = remainder.len() - 1;
        let shift = rem_deg - den_deg;
        let factor = remainder[0] / den_lead;
        let q_idx = quotient.len() - 1 - shift;
        quotient[q_idx] = factor;

        for (i, &c) in den.coeffs.iter().enumerate() {
            remainder[i] -= factor * c;
        }
        remainder.remove(0);

        if remainder.is_empty() {
            break;
        }
    }

    let quotient = Polynomial::trim(quotient);
    let remainder = if remainder.is_empty() {
        Polynomial::zero()
    } else {
        Polynomial::trim(remainder)
    };
    Ok((quotient, remainder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn construction_trims_leading_zeros() {
        let p = Polynomial::new(vec![0.0, 0.0, 1.0, 2.0]).unwrap();
        assert_eq!(p.coeffs(), &[1.0, 2.0]);
        assert_eq!(p.degree(), 1);
    }

    #[test]
    fn all_zeros_normalize_to_single_zero() {
        let p = Polynomial::new(vec![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(p.coeffs(), &[0.0]);
        assert!(p.is_zero());
    }

    #[test]
    fn empty_coeffs_rejected() {
        assert_eq!(
            Polynomial::new(vec![]).unwrap_err(),
            CoreError::EmptyCoefficients
        );
    }

    #[test]
    fn horner_eval_real() {
        // P(s) = s^3 + 2s^2 + 3s + 4
        let p = Polynomial::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_relative_eq!(p.eval(2.0), 26.0);
        assert_relative_eq!(p.eval(-1.0), 2.0);
        assert_relative_eq!(p.eval(0.0), 4.0);
    }

    #[test]
    fn horner_eval_complex_at_imag_axis() {
        // P(s) = s^2 + 1, P(j) = 0
        let p = Polynomial::new(vec![1.0, 0.0, 1.0]).unwrap();
        let v = p.eval_complex(Complex64::new(0.0, 1.0));
        assert_relative_eq!(v.re, 0.0);
        assert_relative_eq!(v.im, 0.0);
    }

    #[test]
    fn derivative_of_cubic() {
        // d/ds(s^3 + 2s^2 + 3s + 4) = 3s^2 + 4s + 3
        let p = Polynomial::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(p.derivative().coeffs(), &[3.0, 4.0, 3.0]);
    }

    #[test]
    fn derivative_of_constant_is_zero() {
        let p = Polynomial::new(vec![5.0]).unwrap();
        assert!(p.derivative().is_zero());
    }

    #[test]
    fn add_unequal_degree() {
        // (s^2 + 1) + (s + 2) = s^2 + s + 3
        let a = Polynomial::new(vec![1.0, 0.0, 1.0]).unwrap();
        let b = Polynomial::new(vec![1.0, 2.0]).unwrap();
        assert_eq!((&a + &b).coeffs(), &[1.0, 1.0, 3.0]);
    }

    #[test]
    fn sub_cancels_leading_term() {
        // (s^2 + s + 1) - (s^2 + 2s + 3) = -s - 2
        let a = Polynomial::new(vec![1.0, 1.0, 1.0]).unwrap();
        let b = Polynomial::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!((&a - &b).coeffs(), &[-1.0, -2.0]);
    }

    #[test]
    fn mul_factored_form() {
        // (s + 1)(s + 2) = s^2 + 3s + 2
        let a = Polynomial::new(vec![1.0, 1.0]).unwrap();
        let b = Polynomial::new(vec![1.0, 2.0]).unwrap();
        assert_eq!((&a * &b).coeffs(), &[1.0, 3.0, 2.0]);
    }

    #[test]
    fn mul_by_zero() {
        let a = Polynomial::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert!((&a * &Polynomial::zero()).is_zero());
    }

    #[test]
    fn mul_scalar() {
        let a = Polynomial::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!((&a * 2.0).coeffs(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn neg_flips_signs() {
        let a = Polynomial::new(vec![1.0, -2.0, 3.0]).unwrap();
        assert_eq!((-a).coeffs(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn div_exact_cubic_minus_one() {
        // (s^3 - 1) / (s - 1) = s^2 + s + 1, remainder 0
        let num = Polynomial::new(vec![1.0, 0.0, 0.0, -1.0]).unwrap();
        let den = Polynomial::new(vec![1.0, -1.0]).unwrap();
        let (q, r) = poly_div(&num, &den).unwrap();
        assert_eq!(q.coeffs(), &[1.0, 1.0, 1.0]);
        assert!(r.is_zero());
    }

    #[test]
    fn div_with_constant_remainder() {
        // (s^2 + 2s + 3) / (s + 1) = s + 1, remainder 2
        let num = Polynomial::new(vec![1.0, 2.0, 3.0]).unwrap();
        let den = Polynomial::new(vec![1.0, 1.0]).unwrap();
        let (q, r) = poly_div(&num, &den).unwrap();
        assert_eq!(q.coeffs(), &[1.0, 1.0]);
        assert_eq!(r.coeffs(), &[2.0]);
    }

    #[test]
    fn div_by_zero_errors() {
        let num = Polynomial::new(vec![1.0, 2.0]).unwrap();
        assert_eq!(
            poly_div(&num, &Polynomial::zero()).unwrap_err(),
            CoreError::DivisionByZero
        );
    }

    #[test]
    fn div_lower_degree_returns_dividend_as_remainder() {
        // s / (s^2 + 1) = 0, remainder s
        let num = Polynomial::new(vec![1.0, 0.0]).unwrap();
        let den = Polynomial::new(vec![1.0, 0.0, 1.0]).unwrap();
        let (q, r) = poly_div(&num, &den).unwrap();
        assert!(q.is_zero());
        assert_eq!(r.coeffs(), &[1.0, 0.0]);
    }

    #[test]
    fn div_quartic_by_quadratic_exact() {
        // (s^4 + 2s^2 + 1) / (s^2 + 1) = s^2 + 1, remainder 0
        let num = Polynomial::new(vec![1.0, 0.0, 2.0, 0.0, 1.0]).unwrap();
        let den = Polynomial::new(vec![1.0, 0.0, 1.0]).unwrap();
        let (q, r) = poly_div(&num, &den).unwrap();
        assert_eq!(q.coeffs(), &[1.0, 0.0, 1.0]);
        assert!(r.is_zero());
    }
}
