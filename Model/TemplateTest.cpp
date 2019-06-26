/*
 * TemplateTest.cpp
 *
 *  Created on: 26/06/2019
 *      Author: Dell User
 */

#include <TemplateTest.h>
namespace niwa{
  TemplateTest::TemplateTest() {
    // TODO Auto-generated constructor stub

  }

  TemplateTest::~TemplateTest() {
    // TODO Auto-generated destructor stub
  }
  template<typename T_>
  T_ TemplateTest::test(T_ val) const {
    T_ return_val = val;
    return return_val;
  }

}
