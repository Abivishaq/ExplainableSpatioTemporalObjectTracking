import matplotlib.pyplot as plt
import streamlit as st

class LogSummary:
    def __init__(self):
        self.predicted_move_frequency = []

    def summarize_log(self, active_log_data):
        """
        Calculates the number of predicted movements for each routine and stores
        it in self.predicted_move_frequency.

        Args:
            active_log_data (dict): Loaded log data with structure:
                {
                    routine_no: (context, pred_n_expl)
                }
        """
        self.predicted_move_frequency = []
        self.true_movement_frequency = []
        for routine_no in sorted(active_log_data.keys()):
            context, pred_n_expl = active_log_data[routine_no]
            # Getting the number of predicted movements for each routine
            num_predictions = len(pred_n_expl)
            self.predicted_move_frequency.append(num_predictions)

            # Getting the number of true movements
            context_edges = context[1]
            context_y_edges = context[2]
            true_movements = (context_y_edges.argmax(-1) != context_edges.argmax(-1)).sum().item()
            self.true_movement_frequency.append(true_movements)


        

        
    
    def plot_frequency_graph(self,true_movement=False):
        if true_movement:
            freq = self.true_movement_frequency
        else:
            freq = self.predicted_move_frequency
        num_points = len(freq)

        fig_width = max(6, num_points * 0.2)
        fig, ax = plt.subplots(figsize=(fig_width, 2))

        # Plot bars
        ax.bar(range(num_points), freq, color='skyblue', width=1.0, align='center')
        ax.set_xlim(0, num_points)

        # Add dense x-axis labels (routine indices)
        ax.set_xticks(range(0,num_points,2))
        ax.set_xticklabels([str(i) for i in range(0,num_points,2)], rotation=90, fontsize=10)

        # Remove y-axis clutter
        ax.set_yticks([])
        ax.set_frame_on(False)

        plt.tight_layout(pad=0.1)
        st.pyplot(fig)
