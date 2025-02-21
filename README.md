
## Table 'Data'

- **PROJECT_ID**: Project code (extra info, I don’t think it’s really necessary)
- **PM**: Responsible management team
- **TASK_ID**: Task code
- **START**: Task start date
- **END**: Theoretical task delivery date (can be used to compare with DELIVERED date to check for delays)
- **TASK_TYPE**: Task type. Some considerations need to be taken into account:
  - **DTP**: Desktop Publishing tasks
  - **Engineering**: Engineering tasks such as file conversions, coding, etc.
  - **LanguageLead**: Linguistic management tasks. Assigned to highly experienced and skilled individuals who work regularly on the project.
  - **Management**: General management tasks.
  - **Miscellaneous**: Various linguistic tasks.
  - **PostEditing**: Post-editing tasks. Similar to Translation tasks, but the TRANSLATOR’s skills are slightly different.
  - **ProofReading**: Full review of a Translation or PostEditing task. This task always follows a Translation or PostEditing. The assigned TRANSLATOR must have more experience than the person who performed the initial step.
  - **Spotcheck**: Partial review of a Translation or PostEditing task. This task always follows a Translation or PostEditing. The assigned TRANSLATOR must have more experience than the person who performed the initial step.
  - **TEST**: Test required to gain access to work with a client. Must be assigned as a priority to the TRANSLATOR with the most experience and quality for the client or subject matter, regardless of cost, but considering the deadline.
  - **Training**: Experience or quality of the TRANSLATOR is not a factor.
  - **Translation**: Translation task. The translator’s quality can be slightly lower than required if the person doing the ProofReading (not Spotcheck) has superior skills. If a Spotcheck is performed, the quality must meet the required standard.
- **SOURCE_LANG**: Source language
- **TARGET_LANG**: Target language
- **TRANSLATOR**: Translator responsible for the task
- **ASSIGNED**: Assignment time (notice) to the TRANSLATOR (see Kanban system: [https://en.wikipedia.org/wiki/Kanban](https://en.wikipedia.org/wiki/Kanban))
- **READY**: Time the TRANSLATOR is notified they can start
- **WORKING**: Time the TRANSLATOR begins the task
- **DELIVERED**: Time the TRANSLATOR delivers the task
- **RECEIVED**: Time the PM receives the task
- **CLOSE**: Time the PM marks the task as completed
- **FORECAST**: Hours of dedication
- **HOURLY_RATE**: Hourly rate for the task
- **COST**: Total cost of the task
- **QUALITY_EVALUATION**: Quality control evaluation
- **MANUFACTURER**: Client
- **MANUFACTURER_SECTOR**: Level 1 client categorization
- **MANUFACTURER_INDUSTRY_GROUP**: Level 2 client categorization
- **MANUFACTURER_INDUSTRY**: Level 3 client categorization
- **MANUFACTURER_SUBINDUSTRY**: Level 4 client categorization

## Table 'Schedules'

- **NAME**: Name of the TRANSLATOR
- **START**: Start time of their workday
- **END**: End time of their workday
- **MON**: Works on Monday? (1 yes, 0 no)
- **TUES**: Works on Tuesday? (1 yes, 0 no)
- **WED**: (1 yes, 0 no)
- **THURS**: (1 yes, 0 no)
- **FRI**: (1 yes, 0 no)
- **SAT**: (1 yes, 0 no)
- **SUN**: (1 yes, 0 no)

## Table 'Clients'

- **CLIENT_NAME**: Client name
- **SELLING_HOURLY_PRICE**: Hourly selling price
- **MIN_QUALITY**: Minimum quality expected from TRANSLATORS
- **WILDCARD**: If all conditions cannot be met, which one can be bypassed.

## Table 'TranslatorsCost+Pairs'

- **TRANSLATOR**: Translator’s name
- **SOURCE_LANG**: Source language
- **TARGET_LANG**: Target language
- **HOURLY_RATE**: Hourly cost rate

## Other Considerations

- The translator’s experience should be assessed based on the hours they have translated for a specific client, client type, or task type.
