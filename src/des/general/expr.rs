//! Port of `src/des/general/expr.ts` — tiny symbolic expression engine.
//!
//! The TS discriminated union `Expr` becomes a Rust `enum`; every algorithm
//! that `switch (e.kind)`-ed becomes a `match`. The algorithm classes
//! (ExprParser, ExprEvaluator, ExprPrinter, ExprDifferentiator, ExprSimplifier,
//! numerical-derivative transforms) become structs implementing `Transform`.
//! Variant constructors (`num`, `add`, …) are free helper fns.

use std::collections::HashMap;

use crate::des::shared::transform::Transform;

// -----------------------------------------------------------------------------
// AST
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl BinOp {
    pub fn symbol(self) -> char {
        match self {
            BinOp::Add => '+',
            BinOp::Sub => '-',
            BinOp::Mul => '*',
            BinOp::Div => '/',
            BinOp::Pow => '^',
        }
    }

    pub fn precedence(self) -> i32 {
        match self {
            BinOp::Add | BinOp::Sub => 1,
            BinOp::Mul | BinOp::Div => 2,
            BinOp::Pow => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuncName {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Abs,
}

impl FuncName {
    pub fn name(self) -> &'static str {
        match self {
            FuncName::Sin => "sin",
            FuncName::Cos => "cos",
            FuncName::Tan => "tan",
            FuncName::Asin => "asin",
            FuncName::Acos => "acos",
            FuncName::Atan => "atan",
            FuncName::Sinh => "sinh",
            FuncName::Cosh => "cosh",
            FuncName::Tanh => "tanh",
            FuncName::Exp => "exp",
            FuncName::Log => "log",
            FuncName::Sqrt => "sqrt",
            FuncName::Abs => "abs",
        }
    }

    pub fn parse(s: &str) -> Option<FuncName> {
        Some(match s {
            "sin" => FuncName::Sin,
            "cos" => FuncName::Cos,
            "tan" => FuncName::Tan,
            "asin" => FuncName::Asin,
            "acos" => FuncName::Acos,
            "atan" => FuncName::Atan,
            "sinh" => FuncName::Sinh,
            "cosh" => FuncName::Cosh,
            "tanh" => FuncName::Tanh,
            "exp" => FuncName::Exp,
            "log" | "ln" => FuncName::Log,
            "sqrt" => FuncName::Sqrt,
            "abs" => FuncName::Abs,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Bin {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Neg(Box<Expr>),
    Func {
        name: FuncName,
        arg: Box<Expr>,
    },
}

// -----------------------------------------------------------------------------
// Variant constructors
// -----------------------------------------------------------------------------

pub fn num(v: f64) -> Expr {
    Expr::Num(v)
}
pub fn var(name: &str) -> Expr {
    Expr::Var(name.to_string())
}
pub fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Bin {
        op,
        left: Box::new(a),
        right: Box::new(b),
    }
}
pub fn add(a: Expr, b: Expr) -> Expr {
    bin(BinOp::Add, a, b)
}
pub fn sub(a: Expr, b: Expr) -> Expr {
    bin(BinOp::Sub, a, b)
}
pub fn mul(a: Expr, b: Expr) -> Expr {
    bin(BinOp::Mul, a, b)
}
pub fn div(a: Expr, b: Expr) -> Expr {
    bin(BinOp::Div, a, b)
}
pub fn pow(a: Expr, b: Expr) -> Expr {
    bin(BinOp::Pow, a, b)
}
pub fn neg(a: Expr) -> Expr {
    Expr::Neg(Box::new(a))
}
pub fn func(name: FuncName, a: Expr) -> Expr {
    Expr::Func {
        name,
        arg: Box::new(a),
    }
}
pub fn zero() -> Expr {
    Expr::Num(0.0)
}
pub fn one() -> Expr {
    Expr::Num(1.0)
}

/// Evaluate a built-in unary math function by name.
pub struct MathFn;

impl MathFn {
    pub fn apply(name: FuncName, x: f64) -> f64 {
        match name {
            FuncName::Sin => x.sin(),
            FuncName::Cos => x.cos(),
            FuncName::Tan => x.tan(),
            FuncName::Asin => x.asin(),
            FuncName::Acos => x.acos(),
            FuncName::Atan => x.atan(),
            FuncName::Sinh => x.sinh(),
            FuncName::Cosh => x.cosh(),
            FuncName::Tanh => x.tanh(),
            FuncName::Exp => x.exp(),
            FuncName::Log => x.ln(),
            FuncName::Sqrt => x.sqrt(),
            FuncName::Abs => x.abs(),
        }
    }
}

// -----------------------------------------------------------------------------
// Parser (recursive descent, precedence climbing).
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum TokKind {
    Num,
    Op,
    LParen,
    RParen,
    Ident,
    Comma,
    Eof,
}

#[derive(Clone, Debug)]
struct Token {
    kind: TokKind,
    text: String,
}

/// Stateful recursive-descent parser (string → Expr).
pub struct ExprParser {
    chars: Vec<char>,
    i: usize,
    cur: Token,
}

impl Default for ExprParser {
    fn default() -> Self {
        ExprParser {
            chars: Vec::new(),
            i: 0,
            cur: Token {
                kind: TokKind::Eof,
                text: String::new(),
            },
        }
    }
}

impl ExprParser {
    pub fn new() -> Self {
        Self::default()
    }

    fn skip_ws(&mut self) {
        while self.i < self.chars.len() && self.chars[self.i].is_whitespace() {
            self.i += 1;
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_ws();
        if self.i >= self.chars.len() {
            return Token {
                kind: TokKind::Eof,
                text: String::new(),
            };
        }
        let c = self.chars[self.i];
        if c == '(' {
            self.i += 1;
            return Token { kind: TokKind::LParen, text: "(".into() };
        }
        if c == ')' {
            self.i += 1;
            return Token { kind: TokKind::RParen, text: ")".into() };
        }
        if c == ',' {
            self.i += 1;
            return Token { kind: TokKind::Comma, text: ",".into() };
        }
        if "+-*/^".contains(c) {
            self.i += 1;
            return Token { kind: TokKind::Op, text: c.to_string() };
        }
        if c.is_ascii_digit() || c == '.' {
            let start = self.i;
            while self.i < self.chars.len() && (self.chars[self.i].is_ascii_digit() || self.chars[self.i] == '.') {
                self.i += 1;
            }
            if self.i < self.chars.len() && (self.chars[self.i] == 'e' || self.chars[self.i] == 'E') {
                self.i += 1;
                if self.i < self.chars.len() && (self.chars[self.i] == '+' || self.chars[self.i] == '-') {
                    self.i += 1;
                }
                while self.i < self.chars.len() && self.chars[self.i].is_ascii_digit() {
                    self.i += 1;
                }
            }
            let text: String = self.chars[start..self.i].iter().collect();
            return Token { kind: TokKind::Num, text };
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = self.i;
            while self.i < self.chars.len() && (self.chars[self.i].is_ascii_alphanumeric() || self.chars[self.i] == '_') {
                self.i += 1;
            }
            let text: String = self.chars[start..self.i].iter().collect();
            return Token { kind: TokKind::Ident, text };
        }
        panic!("unexpected char '{c}' at position {}", self.i);
    }

    fn advance(&mut self) -> Token {
        let t = self.cur.clone();
        self.cur = self.next_token();
        t
    }

    fn expect(&mut self, kind: TokKind) -> Token {
        if self.cur.kind != kind {
            panic!("expected {:?}, got {:?} \"{}\"", kind, self.cur.kind, self.cur.text);
        }
        self.advance()
    }

    fn parse_expression(&mut self) -> Expr {
        let mut left = self.parse_term();
        while self.cur.kind == TokKind::Op && (self.cur.text == "+" || self.cur.text == "-") {
            let op = if self.advance().text == "+" { BinOp::Add } else { BinOp::Sub };
            let right = self.parse_term();
            left = bin(op, left, right);
        }
        left
    }

    fn parse_term(&mut self) -> Expr {
        let mut left = self.parse_factor();
        while self.cur.kind == TokKind::Op && (self.cur.text == "*" || self.cur.text == "/") {
            let op = if self.advance().text == "*" { BinOp::Mul } else { BinOp::Div };
            let right = self.parse_factor();
            left = bin(op, left, right);
        }
        left
    }

    fn parse_factor(&mut self) -> Expr {
        let left = self.parse_unary();
        if self.cur.kind == TokKind::Op && self.cur.text == "^" {
            self.advance();
            let right = self.parse_factor(); // right-associative
            return bin(BinOp::Pow, left, right);
        }
        left
    }

    fn parse_unary(&mut self) -> Expr {
        if self.cur.kind == TokKind::Op && self.cur.text == "-" {
            self.advance();
            return neg(self.parse_unary());
        }
        if self.cur.kind == TokKind::Op && self.cur.text == "+" {
            self.advance();
            return self.parse_unary();
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Expr {
        if self.cur.kind == TokKind::Num {
            let t = self.advance();
            return Expr::Num(t.text.parse::<f64>().unwrap());
        }
        if self.cur.kind == TokKind::LParen {
            self.advance();
            let e = self.parse_expression();
            self.expect(TokKind::RParen);
            return e;
        }
        if self.cur.kind == TokKind::Ident {
            let t = self.advance();
            if self.cur.kind == TokKind::LParen {
                if let Some(fname) = FuncName::parse(&t.text) {
                    self.advance();
                    let arg = self.parse_expression();
                    self.expect(TokKind::RParen);
                    return func(fname, arg);
                }
            }
            return Expr::Var(t.text);
        }
        panic!("unexpected token {:?} \"{}\"", self.cur.kind, self.cur.text);
    }
}

impl Transform<&str, Expr> for ExprParser {
    fn transform(&self, input: &str) -> Expr {
        // `transform` takes `&self`, but parsing needs mutable scan state; clone
        // a fresh parser so the trait stays pure from the caller's view.
        let mut p = ExprParser::new();
        p.chars = input.chars().collect();
        p.i = 0;
        p.cur = p.next_token();
        let e = p.parse_expression();
        if p.cur.kind != TokKind::Eof {
            panic!("unexpected trailing token \"{}\"", p.cur.text);
        }
        e
    }
}

/// Parse a string into an Expr (convenience for the `parse(src)` TS shim).
pub fn parse(src: &str) -> Expr {
    ExprParser::new().transform(src)
}

// -----------------------------------------------------------------------------
// Evaluation.
// -----------------------------------------------------------------------------

pub type Env = HashMap<String, f64>;

pub struct ExprEvaluator;

impl ExprEvaluator {
    pub fn eval(&self, e: &Expr, env: &Env) -> f64 {
        match e {
            Expr::Num(v) => *v,
            Expr::Var(name) => *env
                .get(name)
                .unwrap_or_else(|| panic!("undefined variable '{name}'")),
            Expr::Neg(arg) => -self.eval(arg, env),
            Expr::Func { name, arg } => MathFn::apply(*name, self.eval(arg, env)),
            Expr::Bin { op, left, right } => {
                let a = self.eval(left, env);
                let b = self.eval(right, env);
                match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Pow => a.powf(b),
                }
            }
        }
    }
}

impl Transform<(&Expr, &Env), f64> for ExprEvaluator {
    fn transform(&self, input: (&Expr, &Env)) -> f64 {
        self.eval(input.0, input.1)
    }
}

pub fn evaluate(e: &Expr, env: &Env) -> f64 {
    ExprEvaluator.eval(e, env)
}

// -----------------------------------------------------------------------------
// Pretty-printer.
// -----------------------------------------------------------------------------

pub struct ExprPrinter;

impl ExprPrinter {
    const NEG_PREC: i32 = 4;

    pub fn print(&self, e: &Expr, parent_prec: i32) -> String {
        match e {
            Expr::Num(v) => {
                if *v < 0.0 {
                    format!("({v})")
                } else if v.fract() == 0.0 {
                    format!("{}", *v as i64)
                } else {
                    format!("{v}")
                }
            }
            Expr::Var(name) => name.clone(),
            Expr::Neg(arg) => {
                let inner = self.print(arg, Self::NEG_PREC);
                if Self::NEG_PREC < parent_prec {
                    format!("(-{inner})")
                } else {
                    format!("-{inner}")
                }
            }
            Expr::Func { name, arg } => format!("{}({})", name.name(), self.print(arg, 0)),
            Expr::Bin { op, left, right } => {
                let p = op.precedence();
                let l = self.print(left, p);
                let r = self.print(right, if *op == BinOp::Pow { p - 1 } else { p + 1 });
                let s = format!("{} {} {}", l, op.symbol(), r);
                if p < parent_prec {
                    format!("({s})")
                } else {
                    s
                }
            }
        }
    }
}

pub fn stringify(e: &Expr) -> String {
    ExprPrinter.print(e, 0)
}

// -----------------------------------------------------------------------------
// Simplification.
// -----------------------------------------------------------------------------

pub struct ExprSimplifier;

impl Transform<Expr, Expr> for ExprSimplifier {
    fn transform(&self, e: Expr) -> Expr {
        self.simplify(&e)
    }
}

impl ExprSimplifier {
    pub fn simplify(&self, e: &Expr) -> Expr {
        match e {
            Expr::Num(_) | Expr::Var(_) => e.clone(),
            Expr::Neg(arg) => {
                let a = self.simplify(arg);
                match a {
                    Expr::Num(v) => num(-v),
                    Expr::Neg(inner) => *inner, // --x = x
                    other => neg(other),
                }
            }
            Expr::Func { name, arg } => {
                let a = self.simplify(arg);
                if let Expr::Num(v) = a {
                    num(MathFn::apply(*name, v))
                } else {
                    func(*name, a)
                }
            }
            Expr::Bin { op, left, right } => {
                let a = self.simplify(left);
                let b = self.simplify(right);
                if let (Expr::Num(av), Expr::Num(bv)) = (&a, &b) {
                    match op {
                        BinOp::Add => return num(av + bv),
                        BinOp::Sub => return num(av - bv),
                        BinOp::Mul => return num(av * bv),
                        BinOp::Div => {
                            if *bv != 0.0 {
                                return num(av / bv);
                            }
                        }
                        BinOp::Pow => return num(av.powf(*bv)),
                    }
                }
                let a_num = if let Expr::Num(v) = &a { Some(*v) } else { None };
                let b_num = if let Expr::Num(v) = &b { Some(*v) } else { None };
                match op {
                    BinOp::Add => {
                        if a_num == Some(0.0) {
                            return b;
                        }
                        if b_num == Some(0.0) {
                            return a;
                        }
                    }
                    BinOp::Sub => {
                        if b_num == Some(0.0) {
                            return a;
                        }
                        if a_num == Some(0.0) {
                            return self.simplify(&neg(b));
                        }
                    }
                    BinOp::Mul => {
                        if a_num == Some(0.0) || b_num == Some(0.0) {
                            return zero();
                        }
                        if a_num == Some(1.0) {
                            return b;
                        }
                        if b_num == Some(1.0) {
                            return a;
                        }
                        if a_num == Some(-1.0) {
                            return self.simplify(&neg(b));
                        }
                        if b_num == Some(-1.0) {
                            return self.simplify(&neg(a));
                        }
                    }
                    BinOp::Div => {
                        if b_num == Some(1.0) {
                            return a;
                        }
                        if a_num == Some(0.0) {
                            return zero();
                        }
                    }
                    BinOp::Pow => {
                        if b_num == Some(0.0) {
                            return one();
                        }
                        if b_num == Some(1.0) {
                            return a;
                        }
                        if a_num == Some(1.0) {
                            return one();
                        }
                        if a_num == Some(0.0) {
                            if let Some(bv) = b_num {
                                if bv > 0.0 {
                                    return zero();
                                }
                            }
                        }
                    }
                }
                bin(*op, a, b)
            }
        }
    }
}

pub fn simplify(e: &Expr) -> Expr {
    ExprSimplifier.simplify(e)
}

// -----------------------------------------------------------------------------
// Symbolic differentiation.
// -----------------------------------------------------------------------------

pub struct ExprDifferentiator;

impl ExprDifferentiator {
    pub fn diff(&self, e: &Expr, var_name: &str) -> Expr {
        ExprSimplifier.simplify(&self.diff_raw(e, var_name))
    }

    fn diff_raw(&self, e: &Expr, x: &str) -> Expr {
        match e {
            Expr::Num(_) => zero(),
            Expr::Var(name) => {
                if name == x {
                    one()
                } else {
                    zero()
                }
            }
            Expr::Neg(arg) => neg(self.diff_raw(arg, x)),
            Expr::Bin { op, left, right } => {
                let u = left.as_ref();
                let w = right.as_ref();
                let du = self.diff_raw(u, x);
                let dw = self.diff_raw(w, x);
                match op {
                    BinOp::Add => add(du, dw),
                    BinOp::Sub => sub(du, dw),
                    BinOp::Mul => add(mul(du, w.clone()), mul(u.clone(), dw)),
                    BinOp::Div => div(
                        sub(mul(du, w.clone()), mul(u.clone(), dw)),
                        pow(w.clone(), num(2.0)),
                    ),
                    BinOp::Pow => {
                        if let Expr::Num(wv) = w {
                            mul(mul(num(*wv), pow(u.clone(), num(wv - 1.0))), du)
                        } else {
                            mul(
                                pow(u.clone(), w.clone()),
                                add(
                                    mul(dw, func(FuncName::Log, u.clone())),
                                    mul(w.clone(), div(du, u.clone())),
                                ),
                            )
                        }
                    }
                }
            }
            Expr::Func { name, arg } => {
                let u = arg.as_ref();
                let du = self.diff_raw(u, x);
                match name {
                    FuncName::Sin => mul(func(FuncName::Cos, u.clone()), du),
                    FuncName::Cos => mul(neg(func(FuncName::Sin, u.clone())), du),
                    FuncName::Tan => mul(div(one(), pow(func(FuncName::Cos, u.clone()), num(2.0))), du),
                    FuncName::Asin => mul(div(one(), func(FuncName::Sqrt, sub(one(), pow(u.clone(), num(2.0))))), du),
                    FuncName::Acos => mul(neg(div(one(), func(FuncName::Sqrt, sub(one(), pow(u.clone(), num(2.0)))))), du),
                    FuncName::Atan => mul(div(one(), add(one(), pow(u.clone(), num(2.0)))), du),
                    FuncName::Sinh => mul(func(FuncName::Cosh, u.clone()), du),
                    FuncName::Cosh => mul(func(FuncName::Sinh, u.clone()), du),
                    FuncName::Tanh => mul(sub(one(), pow(func(FuncName::Tanh, u.clone()), num(2.0))), du),
                    FuncName::Exp => mul(func(FuncName::Exp, u.clone()), du),
                    FuncName::Log => mul(div(one(), u.clone()), du),
                    FuncName::Sqrt => mul(div(one(), mul(num(2.0), func(FuncName::Sqrt, u.clone()))), du),
                    FuncName::Abs => mul(div(u.clone(), func(FuncName::Abs, u.clone())), du),
                }
            }
        }
    }
}

pub fn diff(e: &Expr, var_name: &str) -> Expr {
    ExprDifferentiator.diff(e, var_name)
}

// -----------------------------------------------------------------------------
// Numerical derivatives (closures).
// -----------------------------------------------------------------------------

/// Central-difference derivative; O(h²).
pub struct NumericalDerivative {
    pub h: f64,
}

impl Default for NumericalDerivative {
    fn default() -> Self {
        NumericalDerivative { h: 1e-6 }
    }
}

impl NumericalDerivative {
    pub fn eval(&self, f: impl Fn(f64) -> f64, x: f64) -> f64 {
        (f(x + self.h) - f(x - self.h)) / (2.0 * self.h)
    }
}

/// Richardson-extrapolated derivative; O(h⁴).
pub struct RichardsonDerivative {
    pub h: f64,
}

impl Default for RichardsonDerivative {
    fn default() -> Self {
        RichardsonDerivative { h: 1e-3 }
    }
}

impl RichardsonDerivative {
    pub fn eval(&self, f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = self.h;
        let d1 = (f(x + h) - f(x - h)) / (2.0 * h);
        let d2 = (f(x + h / 2.0) - f(x - h / 2.0)) / h;
        (4.0 * d2 - d1) / 3.0
    }
}

/// Central-difference gradient for f: Rⁿ → R.
pub struct NumericalGradient {
    pub h: f64,
}

impl Default for NumericalGradient {
    fn default() -> Self {
        NumericalGradient { h: 1e-6 }
    }
}

impl NumericalGradient {
    pub fn eval(&self, f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
        let h = self.h;
        let n = x.len();
        let mut grad = vec![0.0; n];
        let mut xc = x.to_vec();
        for i in 0..n {
            let orig = xc[i];
            xc[i] = orig + h;
            let fp = f(&xc);
            xc[i] = orig - h;
            let fm = f(&xc);
            xc[i] = orig;
            grad[i] = (fp - fm) / (2.0 * h);
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_eval() {
        let e = parse("x^2 * 3 + 1");
        let mut env = Env::new();
        env.insert("x".to_string(), 2.0);
        assert!((evaluate(&e, &env) - 13.0).abs() < 1e-12);
    }

    #[test]
    fn diff_polynomial() {
        // d/dx (x^3) = 3 x^2 ; at x=2 -> 12
        let e = parse("x^3");
        let d = diff(&e, "x");
        let mut env = Env::new();
        env.insert("x".to_string(), 2.0);
        assert!((evaluate(&d, &env) - 12.0).abs() < 1e-9);
    }

    #[test]
    fn diff_sin() {
        // d/dx sin(x) = cos(x); at 0 -> 1
        let e = parse("sin(x)");
        let d = diff(&e, "x");
        let mut env = Env::new();
        env.insert("x".to_string(), 0.0);
        assert!((evaluate(&d, &env) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn numerical_gradient_matches() {
        // f = x0^2 + x1^2 ; grad at (3,4) = (6,8)
        let g = NumericalGradient::default().eval(|x| x[0] * x[0] + x[1] * x[1], &[3.0, 4.0]);
        assert!((g[0] - 6.0).abs() < 1e-4);
        assert!((g[1] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn simplify_identities() {
        // x * 1 + 0 -> x
        let e = add(mul(var("x"), num(1.0)), num(0.0));
        assert_eq!(stringify(&simplify(&e)), "x");
    }
}
