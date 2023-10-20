from optosim.results.results import Results

for normalise in [False, True]:
    for i in [1, 2]:
        train_run_id = f"mc00{i}0"

        for test_run_id in [f"mc00{i}0", f"mc00{i}1", f"mc00{i}2"]:
            r = Results(
                test_run_id=test_run_id,
                train_run_id=train_run_id,
                nmax=100_000,
                normalise=normalise,
                save_figures=True,
            )

            X_test, y_test, pos_test = r.get_data()
            wa_pred, wa_mse, wa_r2 = r.get_wa_model()
            results = r.get_model_predictions()
            r.get_comparison_model_predictions()

            r.do_plot_mse_scatter()
            r.do_plot_dx_dy_histogram()
            r.do_plot_1d_histogram("dr")
            r.do_plot_1d_histogram("dist")
            r.do_plot_statistics(which="r")
            r.do_plot_statistics(which="ph")
            r.do_plot_2d_hist_with_weights()
