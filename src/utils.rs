#[inline(always)]
pub fn max(a: f64, b: f64) -> f64 {
  if a >= b {
    a
  } else {
    b
  }
}

#[inline(always)]
pub fn exp(a: f64) -> f64 {
  a.exp()
}

#[inline(always)]
pub fn sqrt(a: f64) -> f64 {
  a.sqrt()
}

#[inline(always)]
pub fn fabs(a: f64) -> f64 {
  a.abs()
}

#[inline(always)]
pub fn sel(c: bool, a: f64, b: f64) -> f64 {
  if c {
    a
  } else {
    b
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_max() {
    assert_eq!(10.0, max(10.0, 9.9999));
    assert_eq!(10.0, max(10.0, 10.0));
    assert_eq!(10.0001, max(10.0, 10.0001));
  }

  #[test]
  fn test_exp() {
    assert_eq!((0.0_f64).exp(), exp(0.0));
    assert_eq!((1.0_f64).exp(), exp(1.0));
    assert_eq!((-1.0_f64).exp(), exp(-1.0));
  }

  #[test]
  fn test_sqrt() {
    assert_eq!((0.0_f64).sqrt(), sqrt(0.0));
    assert_eq!((1.0_f64).sqrt(), sqrt(1.0));
    assert_eq!((12.54_f64).sqrt(), sqrt(12.54));
  }

  #[test]
  fn test_fabs() {
    assert_eq!(0.0, fabs(0.0));
    assert_eq!(1.0, fabs(1.0));
    assert_eq!(1.0, fabs(-1.0));
  }

  #[test]
  fn test_sel() {
    assert_eq!(1.0, sel(true, 1.0, 2.0));
    assert_eq!(2.0, sel(false, 1.0, 2.0));
  }
}
