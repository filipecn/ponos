#include <cstdlib> // for EXIT_SUCCESS
#include <cstring> // for strrchr()
#include <iostream>
#include <string>
#include <vector>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

int main(int argc, char *argv[]) {
	try {
		CppUnit::TestFactoryRegistry& registry =
			CppUnit::TestFactoryRegistry::getRegistry();

		CppUnit::TestRunner runner;
		runner.addTest(registry.makeTest());

		CppUnit::TestResult controller;

		CppUnit::TestResultCollector result;
		controller.addListener(&result);

		CppUnit::TextTestProgressListener progress;
		CppUnit::BriefTestProgressListener vProgress;
/*		if (verbose) {
			controller.addListener(&vProgress);
		} else {
			controller.addListener(&progress);
		}

		for (size_t i = 0; i < tests.size(); ++i) {
			runner.run(controller, tests[i]);
		}
*/
		CppUnit::CompilerOutputter outputter(&result, std::cerr);
		outputter.write();

		return result.wasSuccessful() ? EXIT_SUCCESS : EXIT_FAILURE;

	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}
