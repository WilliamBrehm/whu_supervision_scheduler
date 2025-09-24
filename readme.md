# WHU Supervision Scheduler

A constraint programming-based scheduler for assigning exam supervision duties to chairs at WHU (Otto Beisheim School of Management). The scheduler ensures fair distribution of supervision duties while respecting various constraints such as availability and workload balance.

## Features

- **Fair workload distribution** based on configurable share percentages
- **Roundtrip optimization** to minimize days/weeks on campus for selected chairs (so called eco chairs)
- **Constraint handling** for blocked times and fixed assignments
- **Multiple objectives** with configurable weights
- **Comprehensive reporting** including fairness analysis
- **Export** for Excel

## Installation

### Requirements

```bash
pip install pandas numpy ortools openpyxl
```

### Package Structure

```
whu_supervision_scheduler/
├── __init__.py          # Package initialization
├── scheduler.py         # Main scheduler class
├── utils.py            # Utility functions
└── __main__.py         # Command-line interface
```

## Quick Start

### 1. Prepare Input File

Create an Excel file with four sheets:

#### Chairs Sheet
| Chair | Share | Eco |
|-------|-------|--------|
| Ernst | 1.0 | TRUE |
| Shen  | 0.5 | FALSE |
| Spinler | 0.75 | TRUE |

#### Exams Sheet
| Exam | Start | End | Supervisors |
|------|-------|-----|-------------|
| Operations Research | 2025-03-15 09:00 | 2025-03-15 11:00 | 2 |
| Innovation Management | 2025-03-16 14:00 | 2025-03-16 16:30 | 1 |

#### Blocked Sheet (optional)
| Chair | Start | End |
|-------|-------|-----|
| Ernst | 2025-03-16 00:00 | 2025-03-17 00:00 |

#### Fixed Sheet (optional)
| Chair | Exam Id |
|-------|---------|
| Shen  | 0 |

### 2. Basic Usage

#### Command Line
```bash
python -m whu_supervision_scheduler supervision_data.xlsx
```

#### Python Script
```python
from whu_supervision_scheduler import WHUSupervisionScheduler

# Create scheduler
scheduler = WHUSupervisionScheduler()

# Read input file
scheduler.read_supervision_file('supervision_data.xlsx')

# Run scheduling
if scheduler.schedule():
    scheduler.print_solution()
    scheduler.write_solution_to_file('schedule.xlsx')
```

## Advanced Usage

### Customizing Objective Weights

The scheduler balances multiple objectives. You can adjust their relative importance:

```python
scheduler = WHUSupervisionScheduler(
    weight_weeks=40,                    # Minimize weeks for eco chairs
    weight_days=10,                     # Minimize days for eco chairs  
    weight_consecutive_weeks=30,        # Encourage consecutive weeks
    weight_consecutive_days=10,         # Encourage consecutive days
    weight_assignment_fairness=90,      # Ensure fair distribution
    weight_minutes_fairness=1,          # Balance supervision time
    flex_discount=0.15,                 # Discount for flex chairs
    solver_timeout=60                   # Solver time limit in seconds
)
```

### Command Line Options

```bash
python -m whu_supervision_scheduler input.xlsx \
    --output custom_schedule.xlsx \
    --timeout 120 \
    --weight-assignment-fairness 100 \
    --weight-weeks 60 \
    --detailed
```

Available options:
- `-o, --output`: Output filename (default: supervision_schedule.xlsx)
- `-t, --timeout`: Solver timeout in seconds (default: 60)
- `-d, --discount`: Flex chair discount factor (default: 0.15)
- `--weight-*`: Adjust objective weights
- `--detailed`: Export detailed schedule with multiple sheets
- `--validate-only`: Only validate input file without scheduling

### Programmatic Weight Updates

```python
scheduler.update_objective_weights(
    weeks=60,
    assignment_fairness=120
)
```

### Getting Schedule Summary

```python
# Get summary statistics
summary = scheduler.summarize_supervision_info()
print(f"Total supervisions needed: {summary['total_supervisions']}")

# Print formatted summary
scheduler.print_summary()
```

### Exporting Results

```python
from whu_supervision_scheduler import export_detailed_schedule

# Export detailed Excel file
export_detailed_schedule(scheduler.solution, 'detailed_schedule.xlsx')
```

## Understanding the Output

### Fairness Analysis

The solution includes a fairness analysis showing:
- **Share %**: Percentage of total workload assigned to each chair
- **Fair Alloc**: Expected number of supervisions based on share
- **Actual**: Actual number of supervisions assigned
- **Deviation**: Difference between fair and actual allocation
- **Days/Weeks**: Number of unique days/weeks with supervisions
- **Minutes**: Total supervision time and deviation

### Solution Status

- **OPTIMAL**: Best possible solution found
- **FEASIBLE**: Valid solution found (may not be optimal)
- **INFEASIBLE**: No valid solution exists with given constraints

## Algorithm Details

The scheduler uses Google OR-Tools' CP-SAT solver with:

1. **Decision Variables**:
   - Binary assignments (chair-exam pairs)
   - Day/week indicators for eco optimization
   - Consecutive day/week tracking

2. **Constraints**:
   - Supervision requirements
   - No overlapping assignments
   - Blocked times and fixed assignments
   - Time calculation constraints

3. **Multi-objective Optimization**:
   - Minimizes eco chair campus presence
   - Encourages consecutive scheduling
   - Ensures fair workload distribution
   - Balances total supervision time

## Troubleshooting

### No Feasible Solution

If the scheduler reports "INFEASIBLE", check:
1. Blocked times don't conflict with all possible assignments
2. Fixed assignments are compatible to each other
3. Enough chairs available in parallel for peak exam periods

### Poor Solution Quality

To improve solution quality:
1. Increase solver timeout
2. Adjust objective weights based on priorities


## Example

```python
# Complete example with custom configuration
from whu_supervision_scheduler import WHUSupervisionScheduler, create_schedule_summary

# Initialize with custom weights favoring fairness
scheduler = WHUSupervisionScheduler(
    weight_assignment_fairness=150,  # Very high priority on fairness
    weight_weeks=30,                 # Lower priority on week minimization
    flex_discount=0.10,        # Small discount for on-site chairs
    solver_timeout=120               # More time for better solutions
)

# Load and process data
scheduler.read_supervision_file('spring_2025_exams.xlsx')
scheduler.print_summary()

# Solve
if scheduler.schedule():
    # Display results
    scheduler.print_solution()
    
    # Save outputs
    scheduler.write_solution_to_file('spring_2025_schedule.xlsx')
    
    # Create summary report
    summary_df = create_schedule_summary(
        scheduler.solution,
        scheduler.chairs_df,
        scheduler.exams_df
    )
    summary_df.to_excel('spring_2025_summary.xlsx', index=False)
    
    print("\nScheduling completed successfully!")
else:
    print("\nScheduling failed - please check constraints.")
```