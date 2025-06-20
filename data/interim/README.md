
# Information
This merged dataset combines historical project data with current information about translators, schedules, and client information

## Project Identification Columns
| Column | Type | Description |
|--------|------|-------------|
| `PROJECT_ID` | Integer | Unique identifier for each translation project. Used primarily for reference purposes |
| `TASK_ID` | Integer | Unique identifier for specific tasks within a project. A project may contain multiple tasks |

## Management Information
| Column | Type | Description |
|--------|------|-------------|
| `PM` | String | Identifies the project management team responsible for overseeing the task |
| `TASK_TYPE` | String | Categorizes the nature of the task. Different task types require specific translator qualifications as detailed in the original documentation |

## Timeline Information
| Column | Type | Description |
|--------|------|-------------|
| `START_TASK` | Date | The date when the task is scheduled to begin. Used for planning purposes |
| `END_TASK` | Date | The theoretical delivery deadline for the task. Can be compared with `DELIVERED_TASK` date to identify delays |
| `ASSIGNED` | Date | Timestamp when the task was assigned to the translator, serving as initial notice |
| `READY_TASK` | Date | Timestamp when the translator was notified they could begin work on the task |
| `WORKING_TASK` | Date | Timestamp when the translator actually started working on the task |
| `DELIVERED_TASK` | Date | Timestamp when the translator completed and submitted the task |
| `RECEIVED_TASK` | Date | Timestamp when the project manager received the completed task |
| `CLOSE_TASK` | Date | Timestamp when the project manager marked the task as complete |

## Language Pair Information
| Column | Type | Description |
|--------|------|-------------|
| `SOURCE_LANG` | String | The original language of the content being translated |
| `TARGET_LANG` | String | The language into which the content is being translated |
| `PAIR_KEY` | String | A composite key that uniquely identifies the combination of (`TRANSLATOR`, `SOURCE_LANG`, `TARGET_LANG`). Used for efficient lookups and joins |

## Translator Information
| Column | Type | Description |
|--------|------|-------------|
| `TRANSLATOR` | String | Name of the translator assigned to complete the task |
| `QUALITY_EVALUATION` | Integer | Numerical score indicating the quality of the translator's work on this specific task. Used for quality control and can be compared against client requirements |

## Translator Schedule Information
| Column | Type | Description |
|--------|------|-------------|
| `SCHEDULE_START_AVAILABLE` | String | The time of day when the translator begins their workday |
| `SCHEDULE_END_AVAILABLE` | String | The time of day when the translator ends their workday |
| `SCHEDULE_MONDAY` | Integer | Indicates whether the translator works on Mondays |
| `SCHEDULE_TUESDAY` | Integer | Indicates whether the translator works on Tuesdays |
| `SCHEDULE_WEDNESDAY` | Integer | Indicates whether the translator works on Wednesdays |
| `SCHEDULE_THURSDAY` | Integer | Indicates whether the translator works on Thursdays |
| `SCHEDULE_FRIDAY` | Integer | Indicates whether the translator works on Fridays |
| `SCHEDULE_SATURDAY` | Integer | Indicates whether the translator works on Saturdays |
| `SCHEDULE_SUNDAY` | Integer | Indicates whether the translator works on Sundays |

## Financial Information
| Column | Type | Description |
|--------|------|-------------|
| `FORECAST` | Float | Estimated hours required to complete the task. Used for planning and cost estimation |
| `HOURLY_RATE` | Integer | Historical hourly rate paid to the translator for this specific task |
| `COST` | Float | Total cost of the task, typically calculated as `FORECAST` × `HOURLY_RATE` |
| `TRANSLATOR_HOURLY_RATE_LATEST` | Integer | Current hourly rate for the translator for this specific language pair. May differ from the historical `HOURLY_RATE` |
| `DISCREPANCY_HOURLY_RATE` | Integer | The difference between historical and current rates (`HOURLY_RATE` - `TRANSLATOR_HOURLY_RATE_LATEST`). Used to track rate changes over time |
| `CLIENT_HOURLY_PRICE` | Integer | The hourly rate charged to the client. Used to calculate profitability margins |

## Client Information
| Column | Type | Description |
|--------|------|-------------|
| `MANUFACTURER` | String | The name of the client organization |
| `MANUFACTURER_SECTOR` | String | Level 1 categorization of the client's business sector |
| `MANUFACTURER_INDUSTRY_GROUP` | String | Level 2 categorization of the client's industry group |
| `MANUFACTURER_INDUSTRY` | String | Level 3 categorization of the client's specific industry |
| `MANUFACTURER_SUBINDUSTRY` | String | Level 4 categorization of the client's sub-industry. These hierarchical categorizations allow for detailed analysis by business sector |
| `CLIENT_MIN_QUALITY` | Float | The minimum quality score required by the client for translations |
| `CLIENT_WILDCARD` | String | Indicates which requirement can be bypassed if all conditions cannot be met simultaneously |

## Analytical Columns
| Column | Type | Description |
|--------|------|-------------|
| `MEETS_CLIENT_QUALITY` | Boolean | Indicates whether the `QUALITY_EVALUATION` meets or exceeds the `CLIENT_MIN_QUALITY`. Used for quality assurance and compliance reporting |