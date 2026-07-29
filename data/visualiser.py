import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class DataVisualizer:
		def __init__(self, train, test):
				"""Initialize the visualizer with train and test datasets.

				Args:
					train (pd.DataFrame): Training dataset.
					test (pd.DataFrame): Test dataset.
				"""
				self.train = train
				self.test = test

		def get_stats(self, data="train"):
				"""Return a styled table of descriptive statistics.

				Args:
					data (str): Dataset selector.

				Returns:
					pandas.io.formats.style.Styler: Styled statistics table.
				"""
				df = self.train if data == "test" else self.train
				return (
						df.describe()
						.T.style.bar(subset=["mean"], color=px.colors.qualitative.G10[2])
						.background_gradient(subset=["std"], cmap="Blues")
						.background_gradient(subset=["50%"], cmap="BuGn")
				)

		def get_summary(self, data="train"):
				"""Return a styled dataset summary with missing-value metrics.

				Args:
					data (str): Dataset selector.

				Returns:
					pandas.io.formats.style.Styler: Styled summary table.
				"""
				df = self.train if data == "test" else self.train
				sum = pd.DataFrame(df.dtypes, columns=["dtypes"])
				sum["missing#"] = df.isna().sum()
				sum["missing%"] = (df.isna().sum()) / len(df) * 100
				sum["uniques"] = df.nunique().values
				sum["count"] = df.count().values
				# sum['skew'] = df.skew().values
				return sum.style.background_gradient(cmap="Blues")

		def plot_train_test_continuous(self, conts, common_norm=False):
				"""Plot continuous feature distributions for train and test datasets.

				For each column in ``conts``, this creates a KDE comparison between
				train and test data, followed by separate boxplots for each split.

				Args:
						conts (list): Continuous column names to visualize.
						common_norm (bool): Whether to apply common normalization to the
								KDE curves.

				Returns:
						None
				"""
				df = pd.concat(
						[
								self.train[conts].assign(Source="Train"),
								self.test[conts].assign(Source="Test"),
						],
						axis=0,
						ignore_index=True,
				)

				fig, axes = plt.subplots(
						len(conts),
						3,
						figsize=(16, len(conts) * 4.2),
						gridspec_kw={
								"hspace": 0.35,
								"wspace": 0.3,
								"width_ratios": [0.80, 0.20, 0.20],
						},
				)

				for i, col in enumerate(conts):
						ax = axes[i, 0]
						sns.kdeplot(
								data=df[[col, "Source"]],
								x=col,
								hue="Source",
								ax=ax,
								linewidth=2.1,
								common_norm=common_norm,
						)
						ax.set_title(f"\n{col}", fontsize=9, fontweight="bold")
						ax.grid(
								visible=True,
								which="both",
								linestyle="--",
								color="lightgrey",
								linewidth=0.75,
						)
						ax.set(xlabel="", ylabel="")
						ax = axes[i, 1]
						sns.boxplot(
								data=df.loc[df.Source == "Train", [col]],
								y=col,
								width=0.25,
								saturation=0.90,
								linewidth=0.90,
								fliersize=2.25,
								color="#037d97",
								ax=ax,
						)
						ax.set(xlabel="", ylabel="")
						ax.set_title(f"Train", fontsize=9, fontweight="bold")

						ax = axes[i, 2]
						sns.boxplot(
								data=df.loc[df.Source == "Test", [col]],
								y=col,
								width=0.25,
								fliersize=2.25,
								saturation=0.6,
								linewidth=0.90,
								color="#E4591E",
								ax=ax,
						)
						ax.set(xlabel="", ylabel="")
						ax.set_title(f"Test", fontsize=9, fontweight="bold")

				plt.tight_layout()

		def plot_continuous_vs_target(self, conts, target, data='train',common_norm=False):
				"""
				Visualize continuous features against a target variable.

				Args:
					conts (list): Continuous column names to plot.
					target (str): Target column name.
					data (str): Dataset to use ("train" or "test").
					common_norm (bool): Whether to apply common KDE normalization.

				Returns:
					None
				"""
				df = self.train if data == "train" else self.test
				num_cols = len(conts)

				fig, axes = plt.subplots(
						num_cols,
						2,
						figsize=(15, num_cols * 4.5),
						gridspec_kw={"wspace": 0.25, "hspace": 0.35},
				)

				if num_cols == 1:
						axes = [axes]

				for i, col in enumerate(conts):

						ax_kde = axes[i][0]
						sns.kdeplot(
								data=df,
								x=col,
								hue=target,
								ax=ax_kde,
								fill=True,
								alpha=0.3,
								linewidth=2,
								common_norm=common_norm,  # Normalizes each target class independently!
								palette="Set2",
						)
						ax_kde.set_title(
								f"Density of {col} by {target}", fontsize=11, fontweight="bold"
						)
						ax_kde.grid(
								visible=True,
								which="both",
								linestyle="--",
								color="lightgrey",
								linewidth=0.75,
						)
						ax_kde.set(xlabel="", ylabel="Density")

						ax_box = axes[i][1]
						sns.boxplot(
								data=df,
								x=target,
								y=col,
								hue=target,
								ax=ax_box,
								width=0.4,
								fliersize=3,
								saturation=0.8,
								palette="Set2",
								legend=False,
						)
						ax_box.set_title(
								f"Spread of {col} by {target}", fontsize=11, fontweight="bold"
						)
						ax_box.grid(
								visible=True,
								which="both",
								linestyle="--",
								color="lightgrey",
								linewidth=0.75,
						)
						ax_box.set(xlabel=f"Target: {target}", ylabel="")

				plt.tight_layout()
				plt.show()
				
		def plot_categorical_proportions(self, cats):
				"""
				Plot 100% stacked bar charts for categorical feature proportions.

				Args:
					cats (list): Categorical column names to visualize.

				Returns:
					None
				"""
				num_cols = len(cats)

				# Set up a dynamic grid (e.g., 2 charts per row)
				cols_per_row = 2
				rows = (num_cols + cols_per_row - 1) // cols_per_row 

				fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, rows * 4.5), gridspec_kw={'hspace': 0.4})

				# Flatten axes for easy iteration (handles edge cases like 1 row)
				if num_cols > 1:
						axes = axes.flatten()
				else:
						axes = [axes]

				df = pd.concat([self.train[cats].assign(Source = 'Train'), 
														self.test[cats].assign(Source = 'Test')], 
													 axis=0, ignore_index = True);

				for i, col in enumerate(cats):
						ax = axes[i]

						# -----------------------------------------
						# The Math: Calculate proportions by Source
						# -----------------------------------------
						# normalize='index' ensures that Train sums to 1.0 and Test sums to 1.0
						prop_df = pd.crosstab(df['Source'], df[col], normalize='index')

						# Plot the stacked bars directly from the crosstab dataframe
						prop_df.plot(kind='bar', stacked=True, ax=ax, 
												 colormap='Set2', edgecolor='white', linewidth=1.2)

						ax.set_title(f"Distribution of {col}", fontsize=11, fontweight='bold')
						ax.set_ylabel('Proportion (1.0 = 100%)')
						ax.set_xlabel('')

						# Keep the Train/Test labels horizontal for easy reading
						ax.tick_params(axis='x', rotation=0) 

						# Move the legend outside the bar chart so it doesn't cover the data
						ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

				# Hide any empty subplots if the number of features is odd
				for j in range(i + 1, len(axes)):
						fig.delaxes(axes[j])

				plt.tight_layout()
				plt.show()

		def plot_count(self, cats, target):
				'''
				Plot stacked count-based proportions for categorical columns.

				Args:
					cats (list): Categorical column names to plot.
					target (str): Target column name.

				Returns:
					None
				'''
				num_cols = len(cats)
		
				cols_per_row = 2
				rows = (num_cols + cols_per_row - 1) // cols_per_row
				fig, axes = plt.subplots(rows, cols_per_row, figsize=(17, 4 * rows))
				axes = axes.flatten()
		
				for i, col in enumerate(cats):
						ax = axes[i]
						self.train[col].fillna('Missing')
		
						prop_df = pd.crosstab(self.train[col], self.train[target], normalize='index')
		
						prop_df.plot(kind='bar', stacked=True, ax=ax,
												 colormap='Set2', edgecolor='white', linewidth=1.2)
		
						ax.set_title(f"Distribution of {col} by {target}", fontsize=11, fontweight='bold')
						ax.set_ylabel('Proportion (1.0 = 100%)')
						ax.set_xlabel('')
				for i in range(len(cats), len(axes)):
						axes[i].axis('off')
		
				# fig.suptitle(plotname, fontsize=25, fontweight='bold')
				plt.tight_layout()
				plt.show()