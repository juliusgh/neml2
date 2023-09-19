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
#pragma once

#include "neml2/base/Parser.h"
#include "hit.h"

namespace neml2
{
/**
 * @copydoc neml2::Parser
 *
 * The HITParser parses input files written in the [HIT format](https://github.com/idaholab/hit).
 */
class HITParser : public Parser
{
public:
  HITParser() = default;

  virtual ParameterCollection parse(const std::string & filename,
                                    const std::string & additional_input = "") const override;

  /**
   * @brief Extract parameters for a specific object.
   *
   * @param object The object whose parameters are to be extracted.
   * @return ParameterSet The parameters of the object.
   */
  virtual ParameterSet extract_object_parameters(hit::Node * object) const;

private:
  void extract_parameters(hit::Node * object, ParameterSet & params) const;
  void extract_parameter(hit::Node * node, ParameterSet & params) const;
};

} // namespace neml2
