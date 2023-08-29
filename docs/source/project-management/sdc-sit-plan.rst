.. _sdc-sit-plan:

SDC Software Integration Test (SIT) Plan
========================================

Software Inegration Test Timeline
---------------------------------
====================== =============================================================================================================================
Timeline               Step
====================== =============================================================================================================================
Start of development   SDC creates a SIT template and adds/updates test steps throughout the development process
~4 weeks prior to test SDC begins to finalize test steps
~2 weeks prior to test SDC conducts a dry run of the test procedures, makes any necessary fixes, test step updates
~1 week prior to test  SDC conducts a Test Readiness Review (TRR) to review the final version of the test procedures 
Test                   SDC conducts the SIT
~1 week after test     SDC reviews all failures and proposed fixes, decides on time for re-run of failed steps, and updates requirement test status
====================== =============================================================================================================================

Start of development
^^^^^^^^^^^^^^^^^^^^

Over the course of development, the SDC adds steps to the SIT test procedures. This requires a coordinated effort among all SDC developers to document test steps as new features are introduced. Taking an incremental approach allows for better documentation as more complexity is added throughout the course of development. 

~4 weeks prior to test
^^^^^^^^^^^^^^^^^^^^^^

Finialization of test steps should begin about 4 weeks prior the the scheduled SIT. At this point, most of the features being tested in the SIT should be complete and the test steps should become mostly solidified. Finializing the test steps also requires a group effort among all SDC developers, but should be coordinated by the Test Director or Test Conductor.

~2 weeks prior to test
^^^^^^^^^^^^^^^^^^^^^^

The Test Director or Test Conductor will dry run the test procedure and coordinate with any needed SDC developers. The approach is to fully complete the test procedure flow, not necessarily at one time, with developer support available for clarifications and guidelines. The dry run may include actual application execution, as needed to clarify specific test procedure steps and test case assumptions.

~1 week prior to test
^^^^^^^^^^^^^^^^^^^^^

The Test Readiness Review (TRR) brings all participants together in preparation for the upcoming SIT. The main subjects to be reviewed are:

* SIT readiness from all developers' standpoint
* Objectives and success criteria, with functional requirements to be verified
* Prerequisites, such as data sets, and external systems readiness
* Review test procedures and sequence of events
* Personnel support, infrastructure, and networking resources
* Overall test schedule and run-time coordination logistics

Test
^^^^

The execution of the test is also known as the Run for Record (RFR). This activity takes place during a SIT, when Test Conductors conduct the tests. The RFR approach is characterized by gathering all required information as the test develops. This process is materialized by recording dates, times, and events as they occur during the test in the test procedure(s).
There can be multiple RFR instances. The respective run's dates and times are recorded in the test procedure such that a clear differentiation of events is apparent. All redlines and corrections during an execution are incorporated, approved during a Post-Test Review, and released to be used for subsequent RFRs of the same test procedure.

~1 week after test
^^^^^^^^^^^^^^^^^^

During the Post-Test Review, all completed test procedures with their respective reports and verified requirements are reviewed by all involved elements. All failures and issued problem reports are reviewed. The requirements verification scorecard is updated and presented to show the progression through the SOC L4 requirements verification. The team makes recommendations for a development fix and/or subsequent re-test or future regression testing if feasible.
