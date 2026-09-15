# PPE training orchestration.
import os
import time
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm, trange
from ._numerics import *
from .objectives import move_targets


def _initialize(model, lr, nobj, optimizer, optimizer_c, num_init, path, lr_scheduler, lr_scheduler_c, criterion, num_pred, trainloader, valloader, testloader, device, type=None, num_corr=10, num_minres=100, *, runtime):
    pass
    pass
    pass
    init_loss = [[] for _ in range(nobj)]
    initt_loss = [[] for _ in range(nobj)]
    total_init_loss = []
    pareto_archive = []
    num_tasks = nobj
    total_losses = [0.0 for _ in range(num_tasks)]
    num_samples = 0
    pareto_archive = []
    runtime.synchronize()
    total_startI = time.time()
    for epoch in trange(num_init, desc='Init_epoch'):
        runtime.synchronize()
        t0 = time.time()
        model.train(True)
        for batch_idx, (images, labels) in enumerate(tqdm(trainloader, desc='Batch', leave=False)):
            optimizer.zero_grad()
            images = images.to(device)
            labels = move_targets(labels, device)
            alphat = mgda_optimize(model=model, images=images, labels=labels, criterion=criterion, compute_alpha=compute_alpha, optimizer=optimizer, lr_scheduler=lr_scheduler)
            outs = model(images)
            for i in range(len(outs)):
                total_losses[i] += criterion.task_loss(i, outs[i], labels).item() * images.size(0)
            num_samples += images.size(0)
        train_losses = [tl / num_samples for tl in total_losses]
        val_losses, val_top1s = evaluate(model, valloader, nobj, criterion)
        total_init_loss.append(val_losses)
        if epoch >= 0:
            print(f'[Epoch {epoch}] train/  val:   {val_losses}  ') #val_top1: {val_top1s}
        ckpt_file = f'model_init_{num_init}_{type}.pth'
        if epoch == 0:
            model.save_model(path / ckpt_file)
        snapshot = runtime.archive_point(model, 'initial', val_losses, alphat, epoch=epoch)
        pareto_archive.append((np.array(val_losses), snapshot, alphat))
        pruned_archive = prune_archive(pareto_archive)
        best_val, best_ckpt, best_alpha = min(pruned_archive, key=lambda x: np.mean(x[0]))
        if any((np.array_equal(val_losses, v) for v, _, _ in pruned_archive)):
            save_or_copy(model, best_ckpt, path / ckpt_file)
            pareto_archive = pruned_archive
        else:
            pareto_archive = pruned_archive
        dominated_by_archive = any((is_dominated(val_losses, prev_v) for prev_v, _, _ in pareto_archive if (prev_v != np.array(val_losses)).any()))
        'if (epoch > num_init-2 and dominated_by_archive): #20\n                print("Current validation point is dominated by archive; stopping.")\n                break'
    runtime.synchronize()
    train_time = time.time() - t0
    print(f'Initial time: ', train_time)
    alphat = best_alpha.detach().cpu().numpy() if torch.is_tensor(best_alpha) else np.asarray(best_alpha).copy()
    np.save(path / f'alpha_{num_init}.npy', alphat)
    print('Final validation nondominated archive size:', len(pareto_archive))
    runtime.synchronize()
    total_timeI = time.time() - total_startI
    save_or_copy(model, best_ckpt, path / ckpt_file)
    model.load_model(path / ckpt_file)
    test_ilosses, test_top1s = evaluate(model, testloader, nobj, criterion)
    init_losses = best_val
    print(f'f0_after epoch{num_init - 1}  ', init_losses)
    print('f0-test', test_ilosses)
    for i in range(nobj):
        init_loss[i].append(init_losses[i])
        initt_loss[i].append(test_ilosses[i])
    runtime.record_initial(init_losses, test_ilosses, criterion.names)
    pass
    pass
    file_path = runtime.results_dir
    file_path.mkdir(parents=True, exist_ok=True)
    with open(file_path / f'info_cen_init.txt', 'a') as f:
        f.write('=== Training ===\n')
        f.write(f'Initial training time/ epoch {num_init}: {total_timeI:.2f} s\n\n')
    pass
    runtime.synchronize()
    model.save_model(path / f'model_corr_{num_corr}_{type}_old.pth')
    count = 0
    return {name: value for name, value in locals().items() if name in NAVIGATION_STATE}


NAVIGATION_STATE = ('init_loss', 'initt_loss', 'total_init_loss', 'num_tasks', 'best_val',
                    'alphat', 'init_losses', 'ckpt_file', 'count', 'prev', 'ckpt_cfile',
                    'corr_losses_final', 'alpha_corr_old', 'num_pred')


def _navigate(model, lr, nobj, optimizer, optimizer_c, num_init, path, lr_scheduler, lr_scheduler_c, criterion, num_pred, trainloader, valloader, testloader, device, type=None, num_corr=10, num_minres=100, *, runtime, state):
    init_loss = state['init_loss']
    initt_loss = state['initt_loss']
    total_init_loss = state['total_init_loss']
    num_tasks = state['num_tasks']
    best_val = state['best_val']
    alphat = state['alphat']
    init_losses = state['init_losses']
    ckpt_file = state['ckpt_file']
    count = state['count']
    prev = state.get('prev')
    ckpt_cfile = state.get('ckpt_cfile')
    corr_losses_final = state.get('corr_losses_final', init_losses)
    alpha_corr_old = state.get('alpha_corr_old', alphat)
    num_pred = state['num_pred']
    while runtime.keep_running(count):
        pred_loss = [[] for _ in range(nobj)]
        predt_loss = [[] for _ in range(nobj)]
        corr_loss = [[] for _ in range(nobj)]
        corrt_loss = [[] for _ in range(nobj)]
        preference = []
        preference_all = []
        alphas_all = []
        pred_loss_all = [[] for _ in range(nobj)]
        predt_loss_all = [[] for _ in range(nobj)]
        corr_loss_all = [[] for _ in range(nobj)]
        corrt_loss_all = [[] for _ in range(nobj)]
        print(f'\n🔁 Starting navigation cycle {count + 1}')
        pref = runtime.preference(nobj)
        pred = 0
        total_plosses = [0.0 for _ in range(num_tasks)]
        num_psamples = 0
        backt = False
        cont = 0
        if count == 0:
            prev_v = None
            model.load_model(path / ckpt_file)
        else:
            prev_v = prev
            model.load_model(path / ckpt_cfile)
        pareto_parchive = []
        jacobian_trainiter = iter(trainloader)
        lrt = None
        runtime.synchronize()
        t0_pred = time.time()
        while pred < num_pred:
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            if pred == 0:
                printed_corner = False
                printed_extreme = False
                printed_direction_impossible = False
                printed_step = False
            else:
                model.load_model(path / ckpt_pfile)
                printed_corner = True
                printed_extreme = True
                printed_direction_impossible = True
                printed_step = True
            backtrack = True if pred >= 0 else False
            if backt == False:
                backt = backtrack
            else:
                backt = backt
            lrt = lr if lrt == None else lrt
            res, prev, alpha = predictor_step(model, pref, criterion, count, linear_op_template=HVPLinearOperator(trainloader, model, criterion), trainloader=trainloader, jacobian_trainiter=jacobian_trainiter, prev_v=prev_v, step_size=lrt, n_obj=nobj, maxiter=num_minres, momentum=0.9, printed_corner=printed_corner, printed_extreme=printed_extreme, printed_direction_impossible=printed_direction_impossible, printed_step=printed_step, backtrack=backt)
            if res:
                for images, labels in tqdm(trainloader, desc='Batch', leave=False):
                    images = images.to(device)
                    labels = move_targets(labels, device)
                    outs = model(images)
                    for i in range(len(outs)):
                        total_plosses[i] += criterion.task_loss(i, outs[i], labels).item() * images.size(0)
                    num_psamples += images.size(0)
                pred_losses_ = [tl / num_psamples for tl in total_plosses]
                print('Training Predictor losses: ', pred_losses_)
                if pred > 0:
                    preference.append(pref)
                val_plosses, _ = evaluate(model, valloader, nobj, criterion)
                ckpt_pfile = f'model_pred_{num_pred}_{type}.pth'
                if pred == 0:
                    model.save_model(path / ckpt_pfile)
                snapshot = runtime.archive_point(model, 'predictor', val_plosses, alpha, cycle=count, epoch=pred)
                pareto_parchive.append((np.array(val_plosses), snapshot, alpha))
                pareto_parchive = prune_archive(pareto_parchive, typePred=True, pref=pref)
                if pred > num_pred - 2 and is_dominated(val_plosses, init_losses, pref):
                    print(f'❌change too small, change preference/objectives/stepsize .......... Reverting to previous state')
                    pref = runtime.preference(nobj)
                    lrt = runtime.step_size()
                    if count == 0:
                        prev_v = None
                        model.load_model(path / ckpt_file)
                    else:
                        prev_v = prev
                        model.load_model(path / ckpt_cfile)
                    pred = 0
                    backt = False
                    print(f'🔁 Restarting predictor loop with new p={pref}')
                    continue
                elif count > 0 and pred > num_pred - 2:
                    pred_old = []
                    pred_curr = []
                    for i in range(len(pref)):
                        if pref[i] < 0:
                            pred_old.append(corr_losses_final[i])
                            pred_curr.append(val_plosses[i])
                    if not np.all(np.array(pred_curr) <= np.array(pred_old)):
                        print('No progress made, try increasing or reducing step size or changing preference')
                        print('current corrector', corr_losses_final)
                        pref = runtime.preference(nobj)
                        lrt = runtime.step_size()
                        pred = 0
                        backt = False
                        print(f'🔁 Restarting predictor loop with new p={pref}')
                        continue
                elif count >= 0 and pred > 0:
                    pred_old = []
                    pred_curr = []
                    if count == 0:
                        corr_losses_final = init_losses
                    else:
                        corr_losses_final = corr_losses_final
                    for i in range(len(pref)):
                        if pref[i] < 0:
                            pred_old.append(corr_losses_final[i])
                            pred_curr.append(val_plosses[i])
                    if np.all(np.array(pred_curr) <= np.array(pred_old)):
                        print('Early stopping desired objectives have been minimized')
                        save_or_copy(model, pareto_parchive[-1][1], path / ckpt_pfile)
                        pred_losses = [v for v in pareto_parchive[-1][0]]
                        print(f'[Epoch {pred}] train/val:   {pred_losses}')
                        model.load_model(path / ckpt_pfile)
                        break
                elif pred > num_pred - 2 and any((is_dominated(val_plosses, prev_v, pref) for prev_v, _, _ in pareto_parchive if (prev_v != np.array(val_plosses)).any())):
                    print('change too small, change preference/objectives .......... Reverting to previous state')
                    if count == 0:
                        model.load_model(path / ckpt_file)
                    else:
                        prev_v = prev
                        model.load_model(path / ckpt_cfile)
                    break
                elif pred == num_pred - 1:
                    print('✅ Predictor loop finished all iterations without minimizing desired objectives or being dominated by initial point; stopping at last predictor.')
                    action, pref, max_runs = runtime.action(pref, num_pred)
                    if action == 'change_pref':
                        pred = 0
                        backt = False
                        print(f'🔁 Restarting predictor loop with new p={pref}')
                        continue
                    elif action == 'rerun_predictor':
                        num_pred = max_runs
                        pred = 0
                        backt = False
                        print(f'🔁 Restarting predictor loop with new max_runs={max_runs}')
                        continue
                if any((np.array_equal(val_plosses, v) for v, _, _ in pareto_parchive)):
                    model.save_model(path / ckpt_pfile)
                    pareto_parchive = pareto_parchive
                    model.load_model(path / ckpt_pfile)
                else:
                    pareto_parchive = pareto_parchive
                    model.load_model(path / ckpt_pfile)
                save_or_copy(model, pareto_parchive[-1][1], path / ckpt_pfile)
                model.load_model(path / ckpt_pfile)
                pred_losses = [v for v in pareto_parchive[-1][0]]
                print(f'[Epoch {pred}] train/val:   {pred_losses}')
            else:
                pref = runtime.preference(nobj)
                pred = 0
                backt = False
                print(f'🔁 Restarting predictor loop with new p={pref}')
                continue
            pred += 1
        runtime.synchronize()
        train_pred_time = time.time() - t0_pred
        test_plosses, test_ptop1s = evaluate(model, testloader, nobj, criterion)
        eval_plosses = test_plosses
        pred_losses_old = pred_losses.copy()
        print(f'[Epoch {pred}] train/val:   {pred_losses}')
        print(f'[Epoch {pred}] test losses: {test_plosses}') #, test acc: {test_ptop1s}
        print('alpha_pred', alpha)
        pred_losses_final = pred_losses.copy()
        print(f'Predictor time for preference {pref}: ', train_pred_time)
        print('✅ Predictor loop finished successfully. Proceeding to Corrector phase.')
        best_val_accuracy = 0.0
        total_closses = [0.0 for _ in range(num_tasks)]
        num_csamples = 0
        corr_losses_old = pred_losses.copy()
        eval_closses_old = eval_plosses.copy()
        pareto_carchive = []
        runtime.synchronize()
        for corr in trange(num_corr):
            runtime.synchronize()
            t0_corr = time.time()
            model.train()
            if corr == 0:
                model.load_model(path / ckpt_pfile)
            else:
                model.load_model(path / ckpt_cfile)
            for batch_idx, (images, labels) in enumerate(tqdm(trainloader, desc='Batch', leave=False)):
                optimizer.zero_grad(set_to_none=True)
                images = images.to(device)
                labels = move_targets(labels, device)
                alpha_corr = mgda_optimize(model=model, images=images, labels=labels, criterion=criterion, compute_alpha=compute_alpha, optimizer=optimizer_c, lr_scheduler=lr_scheduler_c)
                outs = model(images)
                for i in range(len(outs)):
                    total_closses[i] += criterion.task_loss(i, outs[i], labels).item() * images.size(0)
                num_csamples += images.size(0)
            corr_losses_ = [tl / num_csamples for tl in total_closses]
            print('Corrector train losses: ', corr_losses_)
            ckpt_cfile = f'model_corr_{num_corr}_{type}.pth'
            if corr == 0:
                model.save_model(path / ckpt_cfile)
            val_closses, _ = evaluate(model, valloader, nobj, criterion)
            snapshot = runtime.archive_point(model, 'corrector', val_closses, alpha_corr, cycle=count, epoch=corr)
            pareto_carchive.append((np.array(val_closses), snapshot, alpha_corr))
            pruned_carchive = prune_archive(pareto_carchive)
            best_val_corr, best_ckpt_corr, best_alpha = min(pareto_carchive, key=lambda x: np.mean(x[0]))
            if any((np.array_equal(val_closses, v) for v, _, _ in pruned_carchive)):
                save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                pareto_carchive = pruned_carchive
            else:
                pareto_carchive = pruned_carchive
            corr_losses1 = best_val_corr
            print(f'[Epoch {corr}] train/val:   {corr_losses1}')
            dominated_by_carchive = is_dominated(val_closses, init_losses)
            if corr > num_corr - 2 and is_dominated(val_closses, pred_losses, pref):
                print('❌ Current corrector validation point is dominated by predictor point; stopping.')
                if is_dominated(best_val_corr, pred_losses, pref):
                    print('✅ Best corrector validation point is dominated by predictor; stopping at predictor.')
                    corr_losses1 = pred_losses.copy()
                    alpha_corr = alpha
                    print(f'[Epoch {corr}] train/val: {corr_losses1}')
                    print('alpha_corr reset to alpha_pred: ', alpha_corr)
                    model.load_model(path / ckpt_pfile)
                    model.save_model(path / ckpt_cfile)
                elif is_dominated(best_val, best_val_corr):
                    dm_d = runtime.accept_front()
                    if dm_d:
                        save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                        model.load_model(path / ckpt_cfile)
                        corr_losses1 = best_val_corr
                        init_losses = best_val_corr
                        alpha_corr = best_alpha
                        print('alpha_corr ', alpha_corr)
                        print(f'[Epoch {corr}] train/val: {corr_losses1}')
                    else:
                        print('❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.')
                        corr_losses1 = corr_losses_final.copy()
                        model.load_model(path / f'model_corr_{num_corr}_{type}_old.pth')
                        alpha_corr = alpha_corr_old
                        print('alpha_corr reset to previous corrector alpha: ', alpha_corr)
                        print(f'[Epoch {corr}] train/val: {corr_losses1}')
                        model.save_model(path / ckpt_cfile)
                        cont = 1
                else:
                    print('✅ Best corrector validation point is non-dominated; stopping at best corrector.')
                    save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                    model.load_model(path / ckpt_cfile)
                    corr_losses1 = best_val_corr
                    alpha_corr = best_alpha
                    print('alpha_corr ', alpha_corr)
                    print(f'[Epoch {corr}] train/val: {corr_losses1}')
                break
            elif corr > num_corr - 2 and any((is_dominated(val_closses, prev_cv, pref) for prev_cv, _, _ in pareto_carchive if (prev_cv != np.array(val_closses)).any())):
                print('❌ Current corrector validation point is dominated by corrector archive; stopping at  previous/best corrector.')
                if is_dominated(best_val, best_val_corr):
                    dm_d = runtime.accept_front()
                    if dm_d:
                        save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                        model.load_model(path / ckpt_cfile)
                        corr_losses1 = best_val_corr
                        init_losses = best_val_corr
                        alpha_corr = best_alpha
                        print('alpha_corr ', alpha_corr)
                        print(f'[Epoch {corr}] train/val: {corr_losses1}')
                    else:
                        print('❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.')
                        corr_losses1 = corr_losses_final.copy()
                        model.load_model(path / f'model_corr_{num_corr}_{type}_old.pth')
                        alpha_corr = alpha_corr_old
                        print('alpha_corr reset to previous corrector alpha: ', alpha_corr)
                        print(f'[Epoch {corr}] train/val: {corr_losses1}')
                        model.save_model(path / ckpt_cfile)
                        cont = 1
                elif is_dominated(best_val_corr, pred_losses):
                    print('✅ Best corrector validation point is dominated by predictor; stopping at predictor.')
                    corr_losses1 = pred_losses.copy()
                    print(f'[Epoch {corr}] train/val: {corr_losses1}')
                    model.load_model(path / ckpt_pfile)
                    model.save_model(path / ckpt_cfile)
                else:
                    print('✅ Best corrector validation point is non-dominated; stopping at best corrector.')
                    save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                    model.load_model(path / ckpt_cfile)
                    corr_losses1 = best_val_corr
                    alpha_corr = best_alpha
                    print('alpha_corr ', alpha_corr)
                    print(f'[Epoch {corr}] train/val: {corr_losses1}')
                break
            elif corr > num_corr - 2 and dominated_by_carchive:
                print('Current corrector validation point is dominated by initial point; stopping.')
                if is_dominated(best_val, best_val_corr):
                    dm_d = runtime.accept_front()
                    if dm_d:
                        save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                        model.load_model(path / ckpt_cfile)
                        corr_losses1 = best_val_corr
                        init_losses = best_val_corr
                        alpha_corr = best_alpha
                        print('alpha_corr ', alpha_corr)
                        print(f'[Epoch {corr}] train/val: {corr_losses1}')
                    else:
                        print('❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.')
                        corr_losses1 = corr_losses_final.copy()
                        model.load_model(path / f'model_corr_{num_corr}_{type}_old.pth')
                        alpha_corr = alpha_corr_old
                        print('alpha_corr reset to previous corrector alpha: ', alpha_corr)
                        print(f'[Epoch {corr}] train/val: {corr_losses1}')
                        model.save_model(path / ckpt_cfile)
                        cont = 1
                elif is_dominated(best_val_corr, pred_losses, pref):
                    print('✅ Best corrector validation point is dominated by predictor; stopping at predictor.')
                    corr_losses1 = pred_losses.copy()
                    alpha_corr = alpha
                    print('alpha_corr reset to alpha_pred: ', alpha_corr)
                    print(f'[Epoch {corr}] train/val: {corr_losses1}')
                    model.load_model(path / ckpt_pfile)
                    model.save_model(path / ckpt_cfile)
                else:
                    print('✅ Best corrector validation point is non-dominated; stopping at best corrector.')
                    save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                    model.load_model(path / ckpt_cfile)
                    corr_losses1 = best_val_corr
                    alpha_corr = best_alpha
                    print('alpha_corr ', alpha_corr)
                    print(f'[Epoch {corr}] train/val: {corr_losses1}')
                break
            elif count >= 0 and corr > 0:
                if np.all(np.array(best_val_corr) <= np.array(pred_losses_final)):
                    print('Early stopping desired objectives have all been minimized')
                    if is_dominated(best_val, best_val_corr):
                        dm_d = runtime.accept_front()
                        if dm_d:
                            save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                            model.load_model(path / ckpt_cfile)
                            corr_losses1 = best_val_corr
                            init_losses = best_val_corr
                            alpha_corr = best_alpha
                            print('alpha_corr ', alpha_corr)
                            print(f'[Epoch {corr}] train/val: {corr_losses1}')
                        else:
                            print('❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.')
                            corr_losses1 = corr_losses_final.copy()
                            model.load_model(path / f'model_corr_{num_corr}_{type}_old.pth')
                            alpha_corr = alpha_corr_old
                            print('alpha_corr reset to previous corrector alpha: ', alpha_corr)
                            print(f'[Epoch {corr}] train/val: {corr_losses1}')
                            model.save_model(path / ckpt_cfile)
                            cont = 1
                    else:
                        print('✅ Best corrector validation point is non-dominated; stopping at best corrector.')
                        save_or_copy(model, best_ckpt_corr, path / ckpt_cfile)
                        corr_losses1 = best_val_corr
                        alpha_corr = best_alpha
                        print('alpha_corr ', alpha_corr)
                        print(f'[Epoch {pred}] train/val:   {corr_losses1}')
                    break
        runtime.synchronize()
        train_corr_time = time.time() - t0_corr
        if cont == 1:
            continue
        model.load_model(path / ckpt_cfile)
        test_closses, test_ctop1s = evaluate(model, testloader, nobj, criterion)
        eval_closses = test_closses
        print(f'[Epoch {corr}] train/val:   {corr_losses1}')
        print(f'[Epoch {corr}] test losses: {test_closses}') #, test acc: {test_ctop1s}
        if count == 0 and pred > 0:
            pass
        elif count > 0 and pred > 0:
            pass
        else:
            print('NO Change')
        for i in range(nobj):
            pred_loss_all[i].append(pred_losses[i])
            predt_loss_all[i].append(eval_plosses[i])
        for i in range(nobj):
            corr_loss_all[i].append(corr_losses1[i])
            corrt_loss_all[i].append(eval_closses[i])
        corr_losses_final = corr_losses1.copy()
        model.save_model(path / f'model_corr_{num_corr}_{type}_old.pth')
        alpha_corr_old = alpha_corr
        print(f'corrector time for preference {pref}: ', train_corr_time)
        if not pred == 0:
            pass
            pass
            pass
            for i in range(nobj):
                pred_loss[i].append(pred_loss_all[i][-1])
                corr_loss[i].append(corr_loss_all[i][-1])
                predt_loss[i].append(predt_loss_all[i][-1])
                corrt_loss[i].append(corrt_loss_all[i][-1])
            preference_all.append(preference[-1])
            alphas_all.append(alphat)
            alphas_all.append(alpha_corr)
            initial_point = np.array(init_loss).T
            predictor_point = np.array(pred_loss).T
            corrector_point = np.array(corr_loss).T
            initialt_point = np.array(initt_loss).T
            predictort_point = np.array(predt_loss).T
            correctort_point = np.array(corrt_loss).T
            preference = np.array(preference)
            alphas = np.array([a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a) for a in alphas_all])
            total_init_loss_array = np.array(total_init_loss)
            total_time = train_pred_time + train_corr_time
            print(f'Total time for iteration {count}: ', total_time)
            pass
            file_path = runtime.results_dir
            file_path.mkdir(parents=True, exist_ok=True)
            if count == 0:
                np.save(file_path / f'alpha_{num_init}.npy', alphat)
            with open(file_path / f'first_result_cen{count}.pkl', 'wb') as f:
                pickle.dump((initial_point, predictor_point, corrector_point, preference), f)
            with open(file_path / f'first_result_test_cen{count}.pkl', 'wb') as f:
                pickle.dump((initialt_point, predictort_point, correctort_point, preference), f)
            with open(file_path / f'first_alphas_cen{count}.pkl', 'wb') as f:
                pickle.dump((preference, alphas), f)
            with open(file_path / f'init_total_loss_cen{count}.pkl', 'wb') as f:
                pickle.dump(total_init_loss_array, f)
            with open(file_path / f'info_cen{count}.txt', 'a') as f:
                f.write('=== Training ===\n')
                f.write(f'Predictor time / epoch {num_pred}: {train_pred_time:.2f} s\n\n')
                f.write(f'Corrector time / epoch {num_corr}: {train_corr_time:.2f} s\n\n')
                f.write(f'Training time (total): {total_time:.2f} s\n')
        count += 1
        print('COUNT: ', count)
        runtime.record_cycle(count, pref, pred_losses, corr_losses_final, test_plosses, test_closses)
        runtime.save_boundary({name: value for name, value in locals().items() if name in NAVIGATION_STATE})
        runtime.notify_cycle()
    return runtime.result()


def run_training(*, runtime, **kwargs):
    state = runtime.restore_boundary()
    if state is None or state.get('phase') == 'initialization':
        if state is None:
            runtime.save_boundary({'phase': 'initialization'})
        state = _initialize(runtime=runtime, **kwargs)
        runtime.save_boundary(state)
    runtime.notify_initial()
    return _navigate(runtime=runtime, state=state, **kwargs)
