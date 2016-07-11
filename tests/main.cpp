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

static void
dump(CppUnit::Test* test)
{
    if (test == NULL) {
        std::cerr << "Error: no tests found\n";
        return;
    }

    std::cout << test->getName() << std::endl;
    for (int i = 0; i < test->getChildTestCount(); i++) {
        dump(test->getChildTestAt(i));
    }
}

int main(int argc, char *argv[]) {
	dump(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

	bool verbose = false;
	std::vector<std::string> tests;
	std::cout << "Initiating tests for\n";
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		tests.push_back(argv[i]);
		std::cout << argv[i] << std::endl;
	}
	if (tests.empty())
		return EXIT_SUCCESS;

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
			controller.addListener(&vProgress);

		for (size_t i = 0; i < tests.size(); ++i) {
			runner.run(controller, tests[i]);
		}

		CppUnit::CompilerOutputter outputter(&result, std::cerr);
		outputter.write();

		return result.wasSuccessful() ? EXIT_SUCCESS : EXIT_FAILURE;

	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
