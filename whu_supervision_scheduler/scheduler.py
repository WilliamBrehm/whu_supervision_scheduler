import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import sys
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class WHUSupervisionScheduler:
    """
    A scheduler for assigning exam supervision duties to chairs at WHU.
    
    This class handles reading supervision requirements, creating optimization
    models, and generating fair supervision schedules.
    """
    
    def __init__(self, 
                 weight_weeks: int = 100,
                 weight_days: int = 50,
                 weight_consecutive_weeks: int = 50,
                 weight_consecutive_days: int = 50,
                 weight_assignment_fairness: int = 90,
                 weight_minutes_fairness: int = 1,
                 flex_discount: float = 0.10,
                 solver_timeout: int = 60,
                 eco_priority: int = 10
                 ):
        """
        Initialize the scheduler with objective weights and parameters.
        
        Args:
            weight_weeks: Weight for minimizing week indicators
            weight_days: Weight for minimizing day indicators
            weight_consecutive_weeks: Weight for encouraging consecutive weeks
            weight_consecutive_days: Weight for encouraging consecutive days
            weight_assignment_fairness: Weight for assignment count fairness
            weight_minutes_fairness: Weight for time fairness
            flex_discount: Discount factor for flexible chairs
            solver_timeout: Maximum time in seconds for the solver
        """
        self.weights = {
            'weeks': weight_weeks,
            'days': weight_days,
            'consecutive_weeks': weight_consecutive_weeks,
            'consecutive_days': weight_consecutive_days,
            'assignment_fairness': weight_assignment_fairness,
            'minutes_fairness': weight_minutes_fairness
        }
        self.flex_discount = flex_discount
        self.eco_priority = eco_priority
        self.solver_timeout = solver_timeout
        
        # Data storage
        self.chairs_df = None
        self.exams_df = None
        self.blocked_times_df = None
        self.fixed_exams_df = None
        
        # Model components
        self.model = None
        self.decision_vars = {}
        self.auxiliary_vars = {}
        self.solution = None
        
    def read_supervision_file(self, filename: str) -> None:
        """
        Read chairs and exams data from Excel file.
        
        Args:
            filename: Path to the Excel file containing supervision data
        """
        print(f"Reading supervision data from {filename}...")
        
        # Read the Excel file
        self.chairs_df = pd.read_excel(filename, sheet_name='Chairs')
        self.exams_df = pd.read_excel(filename, sheet_name='Exams')
        self.blocked_times_df = pd.read_excel(filename, sheet_name='Blocked')
        self.fixed_exams_df = pd.read_excel(filename, sheet_name='Fixed')
        
        # Ensure datetime columns are properly formatted
        self.chairs_df['Eco'] = self.chairs_df['Eco'].astype(bool)
        self.exams_df['Start'] = pd.to_datetime(self.exams_df['Start'])
        self.exams_df['End'] = pd.to_datetime(self.exams_df['End'])
        self.blocked_times_df['Start'] = pd.to_datetime(self.blocked_times_df['Start'])
        self.blocked_times_df['End'] = pd.to_datetime(self.blocked_times_df['End'])
        
        print(f"[{len(self.chairs_df)} chairs, {len(self.exams_df)} exams, {len(self.blocked_times_df)} blocked times, {len(self.fixed_exams_df)} fixed exams]")
        
    def update_objective_weights(self, **kwargs) -> None:
        """
        Update objective function weights.
        
        Args:
            **kwargs: Keyword arguments for weights to update
                     (weeks, days, consecutive_weeks, consecutive_days, 
                      assignment_fairness, minutes_fairness)
        """
        for key, value in kwargs.items():
            if key in self.weights:
                self.weights[key] = value
                print(f"Updated weight_{key} to {value}")
            else:
                print(f"Warning: Unknown weight parameter '{key}'")
                
    def summarize_supervision_info(self) -> Dict[str, Any]:
        """
        Summarize the supervision information from loaded data.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.exams_df is None:
            raise ValueError("No exam data loaded. Please run read_supervision_file first.")
            
        summary = {
            'total_chairs': len(self.chairs_df),
            'eco_chairs': self.chairs_df['Eco'].sum(),
            'total_exams': len(self.exams_df),
            'total_supervisions': self.exams_df['Supervisors'].sum(),
            'total_minutes': self._calculate_total_minutes(),
            'exam_weeks': self._summarize_by_week()
        }
        
        return summary
        
    def print_summary(self) -> None:
        """Print a formatted summary of the supervision data."""
        summary = self.summarize_supervision_info()
        
        print("\nSUPERVISION DATA SUMMARY")
        print("=" * 50)
        print(f"Total Chairs: {summary['total_chairs']} ({summary['eco_chairs']} eco)")
        print(f"Total Exams: {summary['total_exams']}")
        print(f"Total Supervisions Needed: {int(summary['total_supervisions'])}")
        print(f"Total Supervision Minutes: {int(summary['total_minutes'])}")
        
        print("\nEXAM DISTRIBUTION BY WEEK")
        print("-" * 50)
        for week_info in summary['exam_weeks']:
            print(f"Week {week_info['week'][1]} of {week_info['week'][0]}: {week_info['total']} exam(s)")
            for day, count in week_info['days'].items():
                print(f"  {day}: {count} exam(s)")
                
    def schedule(self) -> bool:
        """
        Run the complete scheduling process.
        
        Returns:
            True if a feasible solution was found, False otherwise
        """
        if self.exams_df is None:
            raise ValueError("No data loaded. Please run read_supervision_file first.")
            
        print("\nStarting scheduling process...")
        
        # Initialize model
        self._initialize_model()
        
        # Create decision variables
        self._create_decision_variables()
        
        # Calculate auxiliary variables
        self._calculate_auxiliary_variables()
        
        # Add constraints
        self._add_constraints()
        
        # Define objective
        self._define_objective()
        
        # Solve model
        status = self._solve_model()
        
        # Extract solution
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self._extract_solution(status)
            return True
        else:
            self.solution = {'status': 'INFEASIBLE', 'message': 'No feasible solution found'}
            return False
            
    def write_solution_to_file(self, filename: str = 'supervision_schedule.xlsx') -> None:
        """
        Write the solution to an Excel file.
        
        Args:
            filename: Output filename for the schedule
        """
        if self.solution is None or self.solution['status'] == 'INFEASIBLE':
            print("No feasible solution to write.")
            return
            
        # Create assignment matrix
        assignment_data = []
        for chair, exams in self.solution['assignments'].items():
            for exam in exams:
                assignment_data.append({
                    'Supervision Id': exam['supervision_id'],
                    'Exam Name': exam['exam_name'],
                    'Room': exam['room'],
                    'Start': exam['start'],
                    'End': exam['end'],
                    'Chair': chair,
                    'Supervisor': '',
                })
                
        assignment_data.sort(key=lambda x: (x['Start'], x['Supervision Id']))
        
        if assignment_data:
            assignments_df = pd.DataFrame(assignment_data)
            assignments_df.to_excel(filename, index=False)
            print(f"\nSchedule saved to '{filename}'")
            
    def print_solution(self) -> None:
        """Print the solution in a readable format."""
        if self.solution is None:
            print("No solution available.")
            return
            
        if self.solution['status'] == 'INFEASIBLE':
            print("No feasible solution found!")
            return
            
        print("\nFAIRNESS ANALYSIS")
        print("=" * 80)
        
        total_share = self._calculate_total_share()
        total_supervisions = self.exams_df['Supervisors'].sum()
        total_minutes = self._calculate_total_minutes()
        print(f"\nTotal supervisions to assign: {int(total_supervisions)}")
        print("\nChair | Share % | Fair Alloc | Actual | Deviation | Days | Weeks | Fair m. | Actual m. | m. Dev | Eco")
        print("-" * 100)
        
        absolute_deviation = 0
        for _, chair_row in self.chairs_df.iterrows():
            chair_name = chair_row['Chair']
            chair_share = chair_row['Share'] * (1 - self.flex_discount if not chair_row['Eco'] else 1)
            share_percent = (chair_share / total_share) * 100
            fair_alloc = (chair_share / total_share) * total_supervisions
            actual = self.solution['chair_totals'][chair_name]
            deviation = actual - fair_alloc
            absolute_deviation += abs(deviation)
            
            # Compute fair minutes and actual minutes
            fair_minutes = (chair_share / total_share) * total_minutes
            actual_minutes = sum((exam['end'] - exam['start']).total_seconds() // 60 
                               for exam in self.solution['assignments'][chair_name])
            minute_deviation = actual_minutes - fair_minutes
            
            days_assigned = len(self.solution['day_assignments'][chair_name])
            weeks_assigned = len(self.solution['week_assignments'][chair_name])

            eco_status = "Y" if chair_row['Eco'] else "N"

            print(f"{chair_name:12} | {share_percent:6.1f}% | {fair_alloc:10.1f} | "
                  f"{actual:6} | {deviation:+8.1f} | {days_assigned:4} | "
                  f"{weeks_assigned:4} | {fair_minutes:8.1f} | {actual_minutes:8.1f} | "
                  f"{minute_deviation:+8.1f} | {eco_status:3}")
                  
        print(f"\nSolution Status: {self.solution['status']}")
        print(f"Absolute Deviation from Fairness: {absolute_deviation:.1f}")
        print(f"Total Days Assigned: {sum(len(days) for days in self.solution['day_assignments'].values())}")
        print(f"Total Weeks Assigned: {sum(len(weeks) for weeks in self.solution['week_assignments'].values())}")
        
    def visualize_schedule(self) -> None:
        """
        Visualize the supervision schedule using a Gantt chart.
        """
        if self.solution is None or self.solution['status'] == 'INFEASIBLE':
            print("No feasible solution to visualize.")
            return

        # Prepare data for visualization
        schedule_data = []
        for chair, exams in self.solution['assignments'].items():
            for exam in exams:
                schedule_data.append({
                    'Supervision Id': exam['supervision_id'],
                    'Exam Name': exam['exam_name'],
                    'Room': exam['room'],
                    'Start': exam['start'],
                    'End': exam['end'],
                    'Chair': chair,
                })

        schedule_df = pd.DataFrame(schedule_data)

        # Sort data by start time
        schedule_df.sort_values(by='Start', inplace=True)

        # Create Gantt chart
        fig, ax = plt.subplots(figsize=(12, 8))
        chairs = schedule_df['Chair'].unique()
        chair_indices = {chair: idx for idx, chair in enumerate(chairs)}

        for _, row in schedule_df.iterrows():
            # Determine color based on supervision type
            chair_val = row['Chair']
            supervision_id_val = row.get('Supervision Id', None)
            is_fixed = not self.fixed_exams_df[
                (self.fixed_exams_df['Chair'] == chair_val) &
                (self.fixed_exams_df['Supervision Id'] == supervision_id_val)
            ].empty
            if is_fixed:
                c = 'red'
            elif row['Chair'] in self.chairs_df[self.chairs_df['Eco']]['Chair'].values:
                c = 'green'
            else:
                c = 'blue'
            chair_idx = chair_indices[row['Chair']]
            ax.barh(chair_idx, 
                    timedelta(days=1),  # Width of one day
                    left=row['Start'].replace(hour=0, minute=0, second=0, microsecond=0),  # Midnight before start time
                    color=c, 
                    edgecolor=c, 
                    alpha=0.5  # Decreased opacity
                )


        # Add blocked timeframes as black bars
        for _, row in self.blocked_times_df.iterrows():
            chair_idx = chair_indices[row['Chair']]
            ax.barh(chair_idx, 
                (row['End'] - row['Start']),
                left=row['Start'], 
                color='gray',
                edgecolor='gray',
                alpha=0.7)

        # Format the chart
        ax.set_yticks(range(len(chairs)))
        ax.set_yticklabels(chairs)
        
        # Set x-axis to calendar weeks
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('W%U - %Y'))

        # Draw vertical lines for each calendar week
        for week_start in pd.date_range(schedule_df['Start'].min(), schedule_df['End'].max(), freq='W-MON'):
            ax.axvline(week_start, color='gray', linestyle='--', alpha=0.5)

        plt.xticks(rotation=45)
        ax.set_xlabel('Time')
        ax.set_ylabel('Chairs')
        ax.set_title('Supervision Schedule')
        plt.tight_layout()

        # Show the chart
        plt.show()
    
    # Private helper methods
    def _calculate_total_share(self) -> float:
        """Calculate total share considering eco discount."""
        total = 0
        for _, row in self.chairs_df.iterrows():
            if row['Eco']:
                total += row['Share']
            else:
                total += row['Share'] * (1 - self.flex_discount)
        return total
        
    def _calculate_total_minutes(self) -> float:
        """Calculate total supervision minutes needed."""
        total = 0
        for _, exam in self.exams_df.iterrows():
            duration = (exam['End'] - exam['Start']).total_seconds() / 60
            total += duration * exam['Supervisors']
        return total
        
    def _summarize_by_week(self) -> List[Dict]:
        """Summarize exams by week and day."""
        week_summary = {}
        for _, exam in self.exams_df.iterrows():
            week = self._get_exam_week(exam['Start'])
            exam_day = self._get_exam_date(exam['Start'])
            
            if week not in week_summary:
                week_summary[week] = {"total": 0, "days": {}}
            week_summary[week]["total"] += 1
            
            if exam_day not in week_summary[week]["days"]:
                week_summary[week]["days"][exam_day] = 0
            week_summary[week]["days"][exam_day] += 1
            
        result = []
        for week in sorted(week_summary):
            result.append({
                'week': week,
                'total': week_summary[week]["total"],
                'days': dict(sorted(week_summary[week]["days"].items()))
            })
        return result
        
    @staticmethod
    def _get_exam_date(exam_start: datetime) -> datetime.date:
        """Get the date (without time) from exam start datetime."""
        return exam_start.date()
        
    @staticmethod
    def _get_exam_week(exam_start: datetime) -> Tuple[int, int]:
        """Get the week number and year from exam start datetime."""
        return (exam_start.year, exam_start.isocalendar()[1])
        
    @staticmethod
    def _check_exam_overlap(exam1_start: datetime, exam1_end: datetime,
                          exam2_start: datetime, exam2_end: datetime) -> bool:
        """Check if two exams overlap in time."""
        return not (exam1_end + timedelta(minutes=15) <= exam2_start or exam2_end + timedelta(minutes=15) <= exam1_start)
        
    # Model building methods
    def _initialize_model(self) -> None:
        """Initialize the CP-SAT model."""
        self.model = cp_model.CpModel()
        print("CP-SAT model initialized.")
        
    def _create_decision_variables(self) -> None:
        """Create all decision variables for the model."""
        print("Creating decision variables...")
        
        num_chairs = len(self.chairs_df)
        num_exams = len(self.exams_df)
        chair_indices = range(num_chairs)
        exam_indices = range(num_exams)
        
        # Get unique dates and weeks
        exam_dates = sorted(set(self._get_exam_date(exam['Start']) 
                              for _, exam in self.exams_df.iterrows()))
        exam_weeks = sorted(set(self._get_exam_week(exam['Start']) 
                              for _, exam in self.exams_df.iterrows()))
        
        # Store for later use
        self.auxiliary_vars['exam_dates'] = exam_dates
        self.auxiliary_vars['exam_weeks'] = exam_weeks
        self.auxiliary_vars['chair_indices'] = chair_indices
        self.auxiliary_vars['exam_indices'] = exam_indices
        
        # 1. Binary variable: chair i supervises exam j
        x = {}
        for i in chair_indices:
            for j in exam_indices:
                x[i, j] = self.model.NewBoolVar(f'x_{i}_{j}')
        self.decision_vars['x'] = x
        
        # 2. Binary variable: chair i has at least one exam on date d
        y_day = {}
        for i in chair_indices:
            for d_idx, date in enumerate(exam_dates):
                y_day[i, d_idx] = self.model.NewBoolVar(f'y_day_{i}_{d_idx}')
        self.decision_vars['y_day'] = y_day
        
        # 3. Binary variable: chair i has at least one exam in week w
        y_week = {}
        for i in chair_indices:
            for w_idx, week in enumerate(exam_weeks):
                y_week[i, w_idx] = self.model.NewBoolVar(f'y_week_{i}_{w_idx}')
        self.decision_vars['y_week'] = y_week
        
        # 4. Total supervision time variables
        t = {}
        for i in chair_indices:
            t[i] = self.model.NewIntVar(0, int(1e6), f'total_minutes_{i}')
        self.decision_vars['t'] = t

    def _calculate_auxiliary_variables(self) -> None:
        """Calculate auxiliary variables for consecutive days/weeks."""
        print("Calculating auxiliary variables...")
        
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_dates = self.auxiliary_vars['exam_dates']
        exam_weeks = self.auxiliary_vars['exam_weeks']
        y_day = self.decision_vars['y_day']
        y_week = self.decision_vars['y_week']
        
        # Additional decision variables for consecutive days
        z_consecutive_days = {}
        for d_idx in range(len(exam_dates) - 1):
            # Check if the two dates are truly consecutive
            if exam_dates[d_idx + 1] == exam_dates[d_idx] + timedelta(days=1):
                for i in chair_indices:
                    z_consecutive_days[i, d_idx] = self.model.NewBoolVar(f'z_consecutive_day_{i}_{d_idx}')
                    self.model.Add(z_consecutive_days[i, d_idx] <= y_day[i, d_idx])
                    self.model.Add(z_consecutive_days[i, d_idx] <= y_day[i, d_idx + 1])
                    self.model.Add(z_consecutive_days[i, d_idx] >= y_day[i, d_idx] + y_day[i, d_idx + 1] - 1)
        self.decision_vars['z_consecutive_days'] = z_consecutive_days
        
        # Additional decision variables for consecutive weeks
        z_consecutive_weeks = {}
        for w_idx in range(len(exam_weeks) - 1):
            current_week = exam_weeks[w_idx]
            next_week = exam_weeks[w_idx + 1]
            # Convert (year, week) tuple to the Monday of that week
            current_week_date = datetime.fromisocalendar(current_week[0], current_week[1], 1)
            next_week_date = datetime.fromisocalendar(next_week[0], next_week[1], 1)
            # Check if weeks are consecutive
            if next_week_date == current_week_date + timedelta(weeks=1):
                for i in chair_indices:
                    z_consecutive_weeks[i, w_idx] = self.model.NewBoolVar(f'z_consecutive_week_{i}_{w_idx}')
                    self.model.Add(z_consecutive_weeks[i, w_idx] <= y_week[i, w_idx])
                    self.model.Add(z_consecutive_weeks[i, w_idx] <= y_week[i, w_idx + 1])
                    self.model.Add(z_consecutive_weeks[i, w_idx] >= y_week[i, w_idx] + y_week[i, w_idx + 1] - 1)
        self.decision_vars['z_consecutive_weeks'] = z_consecutive_weeks
        
    def _add_constraints(self) -> None:
        """Add all constraints to the model."""
        print("Adding constraints...")
        
        self._add_supervision_requirement_constraints()
        self._add_no_overlap_constraints()
        self._add_day_indicator_constraints()
        self._add_week_indicator_constraints()
        self._add_time_calculation_constraints()
        self._add_fixed_and_blocked_constraints()
        
    def _add_supervision_requirement_constraints(self) -> None:
        """Each exam needs exactly the required number of supervisors."""
        x = self.decision_vars['x']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        
        for j in exam_indices:
            supervisors_needed = int(self.exams_df.iloc[j]['Supervisors'])
            self.model.Add(sum(x[i, j] for i in chair_indices) == supervisors_needed)
            
    def _add_no_overlap_constraints(self) -> None:
        """No overlapping assignments for each chair."""
        x = self.decision_vars['x']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        
        for j1 in exam_indices:
            for j2 in range(j1 + 1, len(exam_indices)):
                exam1 = self.exams_df.iloc[j1]
                exam2 = self.exams_df.iloc[j2]
                if self._check_exam_overlap(exam1['Start'], exam1['End'], 
                                          exam2['Start'], exam2['End']):
                    for i in chair_indices:
                        self.model.Add(x[i, j1] + x[i, j2] <= 1)
                        
    def _add_day_indicator_constraints(self) -> None:
        """Implication constraints for day indicators."""
        x = self.decision_vars['x']
        y_day = self.decision_vars['y_day']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        exam_dates = self.auxiliary_vars['exam_dates']
        
        for d_idx, date in enumerate(exam_dates):
            # Get all exams on this date
            exams_on_date = []
            for j in exam_indices:
                if self._get_exam_date(self.exams_df.iloc[j]['Start']) == date:
                    exams_on_date.append(j)
                    
            if exams_on_date:
                for i in chair_indices:
                    self.model.AddMaxEquality(y_day[i, d_idx], 
                                            [x[i, j] for j in exams_on_date])
                                            
    def _add_week_indicator_constraints(self) -> None:
        """Implication constraints for week indicators."""
        x = self.decision_vars['x']
        y_week = self.decision_vars['y_week']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        exam_weeks = self.auxiliary_vars['exam_weeks']
        
        for w_idx, week in enumerate(exam_weeks):
            # Get all exams in this week
            exams_in_week = []
            for j in exam_indices:
                if self._get_exam_week(self.exams_df.iloc[j]['Start']) == week:
                    exams_in_week.append(j)
                    
            if exams_in_week:
                for i in chair_indices:
                    self.model.AddMaxEquality(y_week[i, w_idx], 
                                            [x[i, j] for j in exams_in_week])
                                            
    def _add_time_calculation_constraints(self) -> None:
        """Total supervision time calculation constraints."""
        x = self.decision_vars['x']
        t = self.decision_vars['t']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        
        for i in chair_indices:
            # Calculate total minutes for all exams assigned to chair i
            exam_minutes = []
            for j in exam_indices:
                start = self.exams_df.iloc[j]['Start']
                end = self.exams_df.iloc[j]['End']
                duration = int((end - start).total_seconds() // 60)
                exam_minutes.append(x[i, j] * duration)
            self.model.Add(t[i] == sum(exam_minutes))
            
    def _add_fixed_and_blocked_constraints(self) -> None:
        """Set assignment for fixed exams and blocked times."""
        x = self.decision_vars['x']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        
        for i in chair_indices:
            for j in exam_indices:
                # Check if this (chair, exam) pair is fixed
                is_fixed = False
                if not self.fixed_exams_df.empty:
                    chair_val = self.chairs_df.iloc[i]['Chair']
                    supervision_id_val = self.exams_df.iloc[j]['Supervision Id'] if 'Supervision Id' in self.exams_df.columns else j
                    fixed_rows = self.fixed_exams_df[
                        (self.fixed_exams_df['Chair'] == chair_val) &
                        (self.fixed_exams_df['Supervision Id'] == supervision_id_val)
                    ]
                    if not fixed_rows.empty:
                        is_fixed = True
                        
                if is_fixed:
                    self.model.Add(x[i, j] == 1)
                    continue
                    
                exam_start = self.exams_df.iloc[j]['Start']
                exam_end = self.exams_df.iloc[j]['End']
                
                # Check if this (chair, exam) pair is blocked due to time overlap
                blocked_overlap = False
                if not self.blocked_times_df.empty:
                    chair_val = self.chairs_df.iloc[i]['Chair']
                    blocked_rows = self.blocked_times_df[self.blocked_times_df['Chair'] == chair_val]
                    for _, blocked_row in blocked_rows.iterrows():
                        blocked_start = blocked_row['Start']
                        blocked_end = blocked_row['End']
                        if not (exam_end <= blocked_start or blocked_end <= exam_start):
                            blocked_overlap = True
                            break
                            
                if blocked_overlap:
                    self.model.Add(x[i, j] == 0)
                    
    def _define_objective(self) -> None:
        """Define the objective function."""
        print("Defining the objective function...")
        
        # Calculate objective components
        day_indicator_sum = self._calculate_day_indicator_sum()
        week_indicator_sum = self._calculate_week_indicator_sum()
        consecutive_days_sum = self._calculate_consecutive_days_sum()
        consecutive_weeks_sum = self._calculate_consecutive_weeks_sum()
        assignment_deviation = self._calculate_assignment_deviation()
        minutes_deviation = self._calculate_minutes_deviation()
        
        # Combined objective with weights
        objective = (
            self.weights['weeks'] * week_indicator_sum 
            + self.weights['days'] * day_indicator_sum 
            - self.weights['consecutive_days'] * consecutive_days_sum
            - self.weights['consecutive_weeks'] * consecutive_weeks_sum
            + self.weights['assignment_fairness'] * assignment_deviation
            + self.weights['minutes_fairness'] * minutes_deviation
        )
        
        self.model.Minimize(objective)
        
    def _calculate_day_indicator_sum(self):
        """Calculate sum of day indicators for eco chairs."""
        y_day = self.decision_vars['y_day']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_dates = self.auxiliary_vars['exam_dates']
        
        return sum(
            y_day[i, d] * (self.eco_priority if self.chairs_df.iloc[i]['Eco'] else 1)
            for i in chair_indices
            for d in range(len(exam_dates))
        )
        
    def _calculate_week_indicator_sum(self):
        """Calculate sum of week indicators for eco chairs."""
        y_week = self.decision_vars['y_week']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_weeks = self.auxiliary_vars['exam_weeks']
        
        return sum(
            y_week[i, w] * (self.eco_priority if self.chairs_df.iloc[i]['Eco'] else 1)
            for i in chair_indices
            for w in range(len(exam_weeks))
        )
        
    def _calculate_consecutive_days_sum(self):
        """Calculate sum of consecutive days for eco chairs."""
        z_consecutive_days = self.decision_vars['z_consecutive_days']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_dates = self.auxiliary_vars['exam_dates']
        
        return sum(
            z_consecutive_days[i, d] * (self.eco_priority if self.chairs_df.iloc[i]['Eco'] else 1)
            for i in chair_indices
            for d in range(len(exam_dates) - 1)
            if (i, d) in z_consecutive_days
        )
        
    def _calculate_consecutive_weeks_sum(self):
        """Calculate sum of consecutive weeks for eco chairs."""
        z_consecutive_weeks = self.decision_vars['z_consecutive_weeks']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_weeks = self.auxiliary_vars['exam_weeks']
        
        return sum(
            z_consecutive_weeks[i, w] * (self.eco_priority if self.chairs_df.iloc[i]['Eco'] else 1)
            for i in chair_indices
            for w in range(len(exam_weeks) - 1)
            if (i, w) in z_consecutive_weeks
        )
  
    def _calculate_assignment_deviation(self):
        """Calculate total assignment count deviation from fairness."""
        x = self.decision_vars['x']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        
        total_share = self._calculate_total_share()
        total_supervisions = self.exams_df['Supervisors'].sum()
        
        total_deviation = 0
        for i in chair_indices:
            chair_share = self.chairs_df.iloc[i]['Share'] * (
                1 - self.flex_discount if not self.chairs_df.iloc[i]['Eco'] else 1
            )
            fair_assignments = int(round((chair_share / total_share) * total_supervisions, 0))
            
            # Actual assignments for chair i
            actual_assignments = sum(x[i, j] for j in exam_indices)
            
            # Create variables for positive and negative deviation
            pos_dev = self.model.NewIntVar(0, int(total_supervisions), f'pos_dev_{i}')
            neg_dev = self.model.NewIntVar(0, int(total_supervisions), f'neg_dev_{i}')
            
            self.model.Add(actual_assignments - fair_assignments == pos_dev - neg_dev)
            
            total_deviation += pos_dev + neg_dev
            
        return total_deviation
        
    def _calculate_minutes_deviation(self):
        """Calculate total time deviation from fairness."""
        t = self.decision_vars['t']
        chair_indices = self.auxiliary_vars['chair_indices']
        
        total_share = self._calculate_total_share()
        total_minutes = self._calculate_total_minutes()
        
        total_deviation = 0
        for i in chair_indices:
            chair_share = self.chairs_df.iloc[i]['Share'] * (
                1 - self.flex_discount if not self.chairs_df.iloc[i]['Eco'] else 1
            )
            fair_minutes = int(round((chair_share / total_share) * total_minutes, 0))
            
            # Actual time for chair i
            actual_time = t[i]
            
            # Create variables for positive and negative time deviation
            pos_minute_dev = self.model.NewIntVar(0, int(1e6), f'pos_minute_dev_{i}')
            neg_minute_dev = self.model.NewIntVar(0, int(1e6), f'neg_minute_dev_{i}')
            
            self.model.Add(actual_time - fair_minutes == pos_minute_dev - neg_minute_dev)
            
            total_deviation += pos_minute_dev + neg_minute_dev
            
        return total_deviation
        
    def _solve_model(self) -> int:
        """Solve the model and return the status."""
        print("Solving the model...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solver_timeout
        status = solver.Solve(self.model)
        self.solver = solver  # Store for later use
        print(f"Solver status: {status}")
        return status
        
    def _extract_solution(self, status: int) -> None:
        """Extract solution from the solved model."""
        print("Extracting solution...")
        
        x = self.decision_vars['x']
        y_day = self.decision_vars['y_day']
        y_week = self.decision_vars['y_week']
        chair_indices = self.auxiliary_vars['chair_indices']
        exam_indices = self.auxiliary_vars['exam_indices']
        exam_dates = self.auxiliary_vars['exam_dates']
        exam_weeks = self.auxiliary_vars['exam_weeks']
        
        self.solution = {
            'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
            'assignments': {},
            'day_assignments': {},
            'week_assignments': {},
            'chair_totals': {},
            'objective_value': self.solver.ObjectiveValue()
        }
        
        # Extract assignments
        for i in chair_indices:
            chair_name = self.chairs_df.iloc[i]['Chair']
            assigned_exams = []
            
            for j in exam_indices:
                if self.solver.Value(x[i, j]) == 1:
                    assigned_exams.append({
                        'supervision_id': self.exams_df.iloc[j].get('Supervision Id', f'Id_{j}'),
                        'exam_name': self.exams_df.iloc[j].get('Exam Name', f'Exam_{j}'),
                        'room': self.exams_df.iloc[j].get('Room', ''),
                        'start': self.exams_df.iloc[j]['Start'],
                        'end': self.exams_df.iloc[j]['End']
                    })
                    
            self.solution['assignments'][chair_name] = assigned_exams
            self.solution['chair_totals'][chair_name] = len(assigned_exams)
            
            # Day assignments
            assigned_days = []
            for d_idx, date in enumerate(exam_dates):
                if self.solver.Value(y_day[i, d_idx]) == 1:
                    assigned_days.append(date)
            self.solution['day_assignments'][chair_name] = assigned_days
            
            # Week assignments
            assigned_weeks = []
            for w_idx, week in enumerate(exam_weeks):
                if self.solver.Value(y_week[i, w_idx]) == 1:
                    assigned_weeks.append(week)
            self.solution['week_assignments'][chair_name] = assigned_weeks