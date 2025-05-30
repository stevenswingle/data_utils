def format_ticks_spines(axs):

    axs.tick_params(labelsize=16,width=2)
    # Make tick labels bold
    for label in axs.get_xticklabels() + axs.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('Arial')
    # Make spines thicker
    axs.spines['left'].set_linewidth(2)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

def jitter_data(input_df, sep=",", x_step=0.5, y_step=0.5, columns=None, x_centre=0, output_file=None):

    """
    Jitters data points to avoid overlap.
    """
    # Read data
    df = input_df.copy()  # Create a copy to avoid modifying the original DataFrame

    # Ensure numeric values in the specified column
    df[columns] = pd.to_numeric(df[columns], errors='coerce')
    df.dropna(subset=[columns], inplace=True)  # Remove rows with NaN in the specified column

    # Initialize x column
    df['x'] = 0.0  # Initialize with zeros

    # Determine bounds for y bands
    ymin = int(df[columns].min() / y_step)
    ymax = int(df[columns].max() / y_step) + 2

    # Process bands
    for iy in range(ymin, ymax):
        # Create an array of Booleans identifying which points lie in the current range
        points_in_range = (df[columns] > (iy * y_step)) & (df[columns] <= (iy + 1) * y_step)

        # Count the number of points in the current range
        num_points = points_in_range.sum()

        if num_points > 1:
            if (num_points % 2) == 0:
                # If there are an even number, create the positive side (e.g., [1, 2])
                a = np.arange(1, (num_points / 2) + 1, 1)
            else:
                # Otherwise, if there are an odd number, create the positive side (e.g., [0, 1, 2])
                a = np.arange(0, int(num_points / 2) + 1, 1)

            # Then the negative side (e.g., [-1, -2])
            b = np.arange(-1, int(num_points / -2) - 1, -1)

            # Now create a new array that can hold both
            c = np.empty((a.size + b.size,), dtype=a.dtype)

            # Interweave them
            c[0::2] = a
            c[1::2] = b

            # Assign jittered x values using .loc
            df.loc[points_in_range, 'x'] = (c * x_step) + x_centre
        else:
            # Assign the center value using .loc
            df.loc[points_in_range, 'x'] = x_centre

    return df
def multi_data_into_dataframe(df_name):
    # Load experimental data and metadata
    df_sample = pd.read_excel(df_name, sheet_name='Results')
    df_meta = pd.read_excel(df_name, sheet_name='Summary_Metadata')

    # Merge the two DataFrames on the common identifier (e.g., 'Experiment ID')
    df_combined = pd.merge(df_sample, df_meta, on='Experiment ID', how='inner')

    # Remove rows where 'Include in Analysis' is 'no'
    df_filtered = df_combined[df_combined["Include in Analysis"] != "no"]

    return df_filtered
def multi_barplot(df, xcol, ycol, filter={}, groupcol=None, jitter='symmetrical',**kwargs):
    """
    Generates a bar plot with optional grouping and filtering.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        xcol (str): The column to use for the x-axis.
        ycol (str): The column to use for the y-axis.
        filter (dict): A dictionary of column-value pairs to filter the DataFrame.
        groupcol (str): The column to use for grouping (optional).
        jitter (str): Type of jitter for scatter points ('random' or 'symmetrical').
    """
    # Check if all dictionary keys exist in df.columns
    missing_keys = [key for key in filter.keys() if key not in df.columns]

    if missing_keys:
        raise KeyError(f"Missing keys in dataframe: {missing_keys}")

    # Apply the general filter
    if filter:
        for key, value in filter.items():
            if isinstance(value, list):  # If the value is a list, use .isin()
                df = df[df[key].isin(value)]
            else:  # Otherwise, filter for exact matches
                df = df[df[key] == value]

    # Dynamically generate bin labels based on unique values in xcol
    bin_labels = df[xcol].unique()  # Ensure labels are sorted
    bin_labels = bin_labels.tolist()  # Convert to list for indexing
    bin_positions = range(len(bin_labels))  # Positions for the bins

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    print(bin_positions, bin_labels)

    if groupcol:
        # Group by the grouping column and xcol
        grouped = df.groupby([groupcol, xcol])[ycol].mean().unstack(groupcol)
        grouped_std = df.groupby([groupcol, xcol])[ycol].std().unstack(groupcol)

        # Plot each group as a separate bar
        bar_width = 0.2  # Width of each bar
        for i, group in enumerate(grouped.columns):
            axs.bar(
                [pos + i * bar_width for pos in bin_positions],  # Offset each group
                grouped[group],
                yerr=grouped_std[group],
                capsize=10,
                width=bar_width,
                label=group,  # Add legend label
                edgecolor='black',
                linewidth=1.5,
                **kwargs  # Pass any additional keyword arguments (e.g., color, alpha, etc.
            )
    else:
        # Calculate mean and standard deviation for each bin
        df_mean = df.groupby(xcol)[ycol].mean()
        df_std = df.groupby(xcol)[ycol].std()

        # Plot the bar chart
        axs.bar(bin_positions, df_mean[bin_labels], yerr=df_std[bin_labels], capsize=10, width=0.5, 
                edgecolor='black', linewidth=2, **kwargs)

    # Set the x-axis tick labels to the bin labels
    axs.set_xticks([pos + (bar_width * (len(grouped.columns) - 1)) / 2 for pos in bin_positions] if groupcol else bin_positions)
    axs.set_xticklabels(bin_labels)

    axs.set_xlabel(xcol, fontname='Arial', fontsize=18, fontweight='bold')
    axs.set_ylabel(ycol, fontname='Arial', fontsize=18, fontweight='bold')
    axs.legend(title=groupcol, fontsize=12) if groupcol else None
    format_ticks_spines(axs)

    # Add jittered scatter points
    if jitter == 'random':
        for i in range(df.shape[0]):
            x = df.iloc[i][xcol]
            x_adj = bin_labels.index(x)  # Find the bin position for the current x value
            axs.scatter(x_adj + np.random.random() * 0.5 - 0.5 / 2, 
                        df.iloc[i][ycol], color='black', alpha=1, s=100)
    elif jitter == 'symmetrical':
        unique_x_values = df[xcol].unique().tolist()
        for i in unique_x_values:
            df_sub_filtered = df[df[xcol] == i]
            df_jitter = jitter_data(
                input_df=df_sub_filtered,
                sep=",",
                columns=ycol,
                x_step=0.25,
                y_step=10.0,
            )
            for j in range(df_sub_filtered.shape[0]):
                x = df_sub_filtered.iloc[j][xcol]
                x_adj = bin_labels.index(x)  # Find the bin position for the current x value
                axs.scatter(x_adj + df_jitter['x'].iloc[j] * 0.5, 
                            df_sub_filtered.iloc[j][ycol], color='black', alpha=1, s=100)
    elif jitter == 'none':
        pass
    return fig, axs