use std::default::Default;

trait Preprocess {
  type Options: Default;

  fn preprocess(&mut self, opt: Self::Options);
}

struct Foo {}

impl Preprocess for Foo {
  type Options = bool;

  fn preprocess(&mut self, opt: bool) {
    if opt {
      // do
    }
  }
}

