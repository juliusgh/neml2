// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/ConstantParameter.h"

namespace neml2
{

template <typename T>
OptionSet
ConstantParameter<T>::expected_options()
{
  OptionSet options = NonlinearParameter<T>::expected_options();
  options.doc() = "A parameter that is just a constant value, generally used to refer to a "
                  "parameter in more than one downstream object.";

  options.set<CrossRef<T>>("value");
  options.set("value").doc() = "The constant value of the parameter";
  return options;
}

template <typename T>
ConstantParameter<T>::ConstantParameter(const OptionSet & options)
  : NonlinearParameter<T>(options),
    _value(this->template declare_parameter<T>("value", "value"))
{
}

template <typename T>
void
ConstantParameter<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    this->_p = _value;

  if (dout_din)
    if (const auto value = this->nl_param("value"))
      this->_p.d(*value) = T::identity_map(this->options());

  // This is zero
  (void)d2out_din2;
}

#define CONSTANTPARAMETER_REGISTER(T)                                                              \
  register_NEML2_object_alias(T##ConstantParameter, #T "ConstantParameter")
FOR_ALL_FIXEDDIMTENSOR(CONSTANTPARAMETER_REGISTER);

} // namespace neml2
