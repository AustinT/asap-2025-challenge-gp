"""Script to train a GP model separately for every task and submit the predictions to Polaris."""

import argparse
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import polaris as po
import tanimoto_gp
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)


def INV_SOFTPLUS(x):
    return jnp.log(jnp.exp(x) - 1.0)


PROP_TO_TRANSFORM_INFO = {
    "MLM": 0.1,
    "LogD": None,
    "MDR1-MDCKII": 1e-3,
    "HLM": 0.1,
    "KSOL": 1.0,
    "pIC50 (MERS-CoV Mpro)": None,
    "pIC50 (SARS-CoV-2 Mpro)": None,
}


def best_tanimoto_gp(smiles_train, y_train, smiles_test) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Tanimoto GPs on ECFP fingerprints with different radii.

    Returns the best GP model (judged by marginal likelihood) and its parameters.

    Returns GP predictions on the training and test sets with optimized hyperparameters.
    """
    r_to_result = dict()
    y_var = np.var(y_train)
    for fp_rad in range(0, 4):
        logger.info(f"Starting GP training for radius {fp_rad}...")
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_rad, fpSize=1
        )  # NOTE: FP size is not used, set to dummy value

        def smiles_to_fp(smiles: str):
            mol = Chem.MolFromSmiles(smiles)
            return mfpgen.GetSparseCountFingerprint(mol)

        # Init GP model to have reasonable hyperparameters
        gp = tanimoto_gp.ZeroMeanTanimotoGP(smiles_to_fp, smiles_train, y_train)
        gp_params = tanimoto_gp.TanimotoGP_Params(
            raw_amplitude=INV_SOFTPLUS(y_var), raw_noise=INV_SOFTPLUS(0.1 * y_var)
        )

        # Set up optimizer
        optimizer = optax.adam(3e-2)
        opt_state = optimizer.init(gp_params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(lambda x: -gp.marginal_log_likelihood(x))(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Run optimization loop
        for opt_idx in range(2_000):  # probably too many steps, but that's ok!
            gp_params, opt_state, loss = step(gp_params, opt_state)
            if opt_idx % 250 == 0:
                logger.debug(f"Step {opt_idx}, {loss=:>15}, {gp_params=}")

        # Make final predictions
        y_pred_train, _ = gp.predict_y(gp_params, smiles_train, full_covar=False)
        y_pred_test, _ = gp.predict_y(gp_params, smiles_test, full_covar=False)
        r_to_result[fp_rad] = (float(loss), gp_params, np.asarray(y_pred_train), np.asarray(y_pred_test))
        logger.info(f"Finished GP training for radius {fp_rad}. Loss: {loss}, {gp_params=}")

    # Find the best GP model
    best_res = min(r_to_result.values(), key=lambda x: x[0])
    return best_res[2], best_res[3]


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("challenge", type=str, choices=["admet", "potency"])
    parser.add_argument("polaris_uname", type=str, help="Polaris username")
    parser.add_argument("--submit", action="store_true", help="Whether to submit to Polaris")
    args = parser.parse_args()

    # Load Polaris competition
    competition = po.load_competition(f"asap-discovery/antiviral-{args.challenge}-2025")
    competition.cache()  # recommended
    train, test = competition.get_train_test_split()
    logger.info(f"Loaded competition: {competition}")

    # Train GPs, make predictions
    y_pred = dict()
    for tgt in train[0][1].keys():
        logger.info(f"### Training GP model for {tgt} ###")

        # Find (smiles, y) pairs for this target
        sm_train, y_train = zip(*train)
        y_train = np.asarray([y[tgt] for y in y_train])
        non_nan_mask = ~np.isnan(y_train)
        sm_train = np.array(sm_train)[non_nan_mask].copy().tolist()
        y_train = np.array(y_train)[non_nan_mask].copy()

        # (potentially) apply log(x+eps) transform to targets
        if PROP_TO_TRANSFORM_INFO[tgt] is not None:
            y_train = np.log(y_train + PROP_TO_TRANSFORM_INFO[tgt])

        # Set GP mean, and set targets to be the residuals
        gp_mean = np.median(y_train)  # median is more robust than mean
        y_train -= gp_mean

        # Summary statistics
        logger.info(
            f"Number of training samples: {len(y_train)}, median: {np.median(y_train)}, "
            f"mean: {np.mean(y_train)}, variance: {np.var(y_train)}"
        )

        # Find the best GP model
        y_pred_train, y_pred_test = best_tanimoto_gp(sm_train, y_train, test)

        # Record train metrics
        print(f"y_pred_train MAE: {np.mean(np.abs(y_pred_train - y_train)):.3g}")
        print(f"GP mean MAE: {np.mean(np.abs(y_train)):.3g}")

        # Undo test transformations
        y_pred_test = (y_pred_test + gp_mean).copy()
        if PROP_TO_TRANSFORM_INFO[tgt] is not None:
            y_pred_test = np.exp(y_pred_test) - PROP_TO_TRANSFORM_INFO[tgt]
        y_pred[tgt] = y_pred_test.copy()  # copy just to be safe

    if args.submit:
        competition.submit_predictions(
            predictions=y_pred,
            prediction_name="tanimoto-kernel-gp",
            prediction_owner=args.polaris_uname,
            report_url="https://github.com/AustinT/asap-2025-challenge-gp",
            # The below metadata is optional, but recommended.
            github_url="https://github.com/AustinT/asap-2025-challenge-gp",
            description="Tanimoto Kernel GP",
        )

    logger.info("END OF MAIN")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.DEBUG)
    main()
