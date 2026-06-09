# This file contains the main training loop for the Preference Pareto Exploration (PPE) framework, including both the predictor and corrector phases. 
# It handles model training, validation, checkpointing, and Pareto archive management for multi-objective optimization in multi-task learning settings.

import matplotlib
matplotlib.use('webAgg') 
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from Data.uci_data import *
from Data.mnist_data import *
from tqdm import tqdm, trange
import torch.nn as nn
from .function import *




def run(model,lr,nobj, optimizer,optimizer_c, num_init, path,lr_scheduler,lr_scheduler_c,criterion,  num_pred,trainloader,valloader,testloader, device, type = None,num_corr= 10, num_minres= 100): #

    # Initialize figure and axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Set up interactive mode
    plt.ion()
    #Initial point
    init_loss = [[] for _ in range(nobj)]
    initt_loss = [[] for _ in range(nobj)]

    total_init_loss = []
    pareto_archive = []   
    num_tasks = nobj
    total_losses = [0.0 for _ in range(num_tasks)]
    num_samples = 0
    
    pareto_archive = []   # list of tuples (val_loss_vector, ckpt_path)

    torch.cuda.synchronize()
    total_startI = time.time()
    if os.path.exists(path / f"model_init_{num_init}_{type}.pth"):  # ########################################### Comment from here to below
        print("Initial model already exist. Computing loss............")
        ckpt_file = f"model_init_{num_init}_{type}.pth"
        model.load_model(path / ckpt_file)

        # VALIDATION evaluation drives archive & early stopping
        best_val, val_top1s = evaluate(model, valloader,nobj, criterion)
        print(f"train/  val:   {best_val}  val_top1: {val_top1s}")
        alphat = np.load(path /f"alpha_{num_init}.npy")

    else:
        for epoch in trange(num_init, desc='Init_epoch'):
            torch.cuda.synchronize()
            t0 = time.time()
            model.train(True)

            for batch_idx, (images, labels) in enumerate(tqdm(trainloader, desc='Batch', leave=False)):
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                alphat = mgda_optimize(model=model,images=images,labels=labels,criterion=criterion,compute_alpha=compute_alpha,optimizer=optimizer,lr_scheduler=lr_scheduler)
                #pmgda_optimize(model,images,labels,trainloader,criterion,np.array([1/nobj]*nobj), optimizer_c,lr_scheduler_c,init_lr=1e-2)
                outs = model(images)
                for i in range(len(outs)):
                    total_losses[i] += criterion(outs[i], labels[:, i].long()).item() * images.size(0)
                num_samples += images.size(0)

            train_losses = [tl / num_samples for tl in total_losses]
            #print("Training initial losses: ", train_losses)

            # VALIDATION evaluation drives archive & early stopping
            val_losses, val_top1s = evaluate(model, valloader, nobj, criterion)
            #if lr_scheduler is not None:
            #    val_loss_scalar = sum(val_losses) / len(val_losses)
            #    lr_scheduler.step(val_loss_scalar)


            total_init_loss.append(val_losses) # store all training validation losses during initialization to observe trend and convergence

            if epoch >= 0 : #and (epoch % 10 == 0 or epoch == num_init - 1):
                #print(f"[Epoch {epoch}] train: {train_losses}")
                print(f"[Epoch {epoch}] train/  val:   {val_losses}  val_top1: {val_top1s}")

            # Save checkpoint for potential inclusion in the archive
            ckpt_file = f"model_init_{num_init}_{type}.pth"
            
            if epoch==0:
                #initialize model file
                model.save_model(path/ckpt_file)

            # Add to archive and prune dominated points
            # Append current checkpoint temporarily to archive
            pareto_archive.append((np.array(val_losses), ckpt_file, alphat))
            # Prune dominated points
            pruned_archive = prune_archive(pareto_archive)
            best_val, best_ckpt ,best_alpha= min(pruned_archive,key=lambda x: np.mean(x[0]))


            if any(np.array_equal(val_losses, v) for v, _, _ in pruned_archive):
                
                # Only save if it survives pruning (nondominated)
                #model.save_model(ckpt_file)
                save_or_copy(model,best_ckpt, path/ckpt_file)
                pareto_archive = pruned_archive  # keep only nondominated points
            else:
                # Current checkpoint is dominated, do not save
                pareto_archive = pruned_archive  # still update archive to remove dominated ones

       
            dominated_by_archive = any(is_dominated(val_losses, prev_v) for prev_v, _,_ in pareto_archive if (prev_v != np.array(val_losses)).any())
            # We don't force stop on a single dominated check; choose your rule. Here we *don't* stop immediately,
            # because later epochs might produce new nondominated points. If you want strict stopping, uncomment:
            """if (epoch > num_init-2 and dominated_by_archive): #20
                print("Current validation point is dominated by archive; stopping.")
                break"""
        
        torch.cuda.synchronize()
        train_time = time.time() - t0

        print(f"Initial time: ", train_time)
        
        # alpha can be numpy OR torch tensor
        if torch.is_tensor(best_alpha):
            alphat = best_alpha.detach().cpu().numpy()
        np.save(path /f"alpha_{num_init}.npy", alphat)
  

        # End training epochs -------------------------------------------------
        # At this point, pareto_archive contains nondominated validation checkpoints.
        print("Final validation nondominated archive size:", len(pareto_archive))


    torch.cuda.synchronize()
    total_timeI = time.time() - total_startI
    #Test the initial optimal point
    # Load best checkpoint
    model.load_model(path/ckpt_file)
    test_ilosses, test_top1s = evaluate(model, testloader, nobj, criterion)
    init_losses = best_val # [v for v in pareto_archive[-1][0]]

    print(f"f0_after epoch{num_init-1}  ",  init_losses)
    print("f0-test", test_ilosses)
        
    for i in range(nobj):
        init_loss[i].append(init_losses[i])
        initt_loss[i].append(test_ilosses[i])
    

    # Plot initial point
    #ax.scatter(init_loss1[-1,0], init_loss2[-1,1], init_loss3[-1,2], label='Initial Point', color='black',s=50)
    ax.scatter(init_loss[0][-1], init_loss[1][-1], init_loss[2][-1], label='Initial Point', color='black',s=50)
    if type == "UCI":
        ax.set_xlabel("age-1", fontsize = 20)
        ax.set_ylabel("education-2", fontsize = 20)
        ax.set_zlabel("marital_stat-3",labelpad=10, fontsize = 20)

    elif type =="INT":
        ax.set_xlabel("fmnist-1", fontsize = 20)
        ax.set_ylabel("kmnist-2", fontsize = 20)
        ax.set_zlabel("mnist-3",labelpad=10, fontsize = 20)

    else:
        ax.set_xlabel("f1", fontsize = 20)
        ax.set_ylabel("f2", fontsize = 20)
        ax.set_zlabel("f3",labelpad=10, fontsize = 20)
    
    try:
        # Works in normal Python scripts
        FILE_DIR = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter/IPython
        FILE_DIR = Path(os.getcwd()).resolve()
    file_path = FILE_DIR.parent / f"Results/{type}"
    file_path.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists
    with open(file_path / f'info_cen_init.txt', "a") as f:
        f.write("=== Training ===\n")
        f.write(f"Initial training time/ epoch {num_init}: {total_timeI:.2f} s\n\n")
       


    # Pause to allow plot to update
    # Display and refresh
    plt.pause(0.5)

    torch.cuda.synchronize()
    count = 0
    while True:
        pred_loss = [[] for _ in range(nobj)]
        predt_loss = [[] for _ in range(nobj)]

        corr_loss = [[] for _ in range(nobj)]
        corrt_loss = [[] for _ in range(nobj)]

        preference = []
        preference_all = []
        alphas_all = []
        #best_val_accuracy =  0.0
        #best_val_loss=  float("inf") #0.0

        pred_loss_all = [[] for _ in range(nobj)]
        predt_loss_all = [[] for _ in range(nobj)]
        corr_loss_all = [[] for _ in range(nobj)]
        corrt_loss_all = [[] for _ in range(nobj)]
        

        print(f"\n🔁 Starting training from epoch {count} (fresh run)")
        pref = get_preference(nobj)
        pred = 0
        total_plosses = [0.0 for _ in range(num_tasks)]
        num_psamples = 0
        backt = False
        cont = 0
        if count == 0:
            prev_v= None
            model.load_model(path / ckpt_file)
            #print("PATH: ", path / ckpt_file)
        else:
            prev_v= prev
            model.load_model(path / ckpt_cfile)
        
        pareto_parchive = []            # stores (val_ploss_vector, filepath)
        jacobian_trainiter = iter(trainloader)
        lrt = None
        torch.cuda.synchronize()
        t0_pred = time.time()
        while pred < num_pred:
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            
            if pred == 0:
                # flags for one-batch print control
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
            
            backtrack = True if pred >= 0 else False #5
            if backt == False:
                backt = backtrack
            else:
                backt = backt

            
            lrt = lr if lrt == None else lrt 

            #print("Backtracking set to ", backt)
            res,prev, alpha = predictor_step(model, pref,criterion,count,linear_op_template = HVPLinearOperator(trainloader,model,criterion),trainloader=trainloader, jacobian_trainiter=jacobian_trainiter,prev_v=prev_v,step_size=lrt, n_obj=nobj,maxiter=num_minres, momentum = 0.9,  printed_corner= printed_corner, #momentum =0.9
                                printed_extreme = printed_extreme, printed_direction_impossible= printed_direction_impossible, printed_step= printed_step, backtrack = backt)
            
            if res:
                for images, labels in tqdm(trainloader, desc='Batch', leave=False):
                    images = images.to(device)
                    labels = labels.to(device)
                    outs = model(images)  
                    
                    for i in range(len(outs)):
                        total_plosses[i] += criterion(outs[i], labels[:, i].long()).item() * images.size(0)
                
                    num_psamples += images.size(0)

                pred_losses_ = [tl / num_psamples for tl in total_plosses]
                print("Training Predictor losses: ", pred_losses_)
                if pred > 0:
                    preference.append(pref)

                # VALIDATION evaluation drives archive & early stopping
                val_plosses, _ = evaluate(model, valloader, nobj, criterion)
                # --- Checkpoint path
                ckpt_pfile = f"model_pred_{num_pred}_{type}.pth"
                if pred == 0:
                    #initialize model file
                    model.save_model(path/ckpt_pfile)
                
            
                # --- Add to Pareto archive and prune dominated points
                pareto_parchive.append((np.array(val_plosses), ckpt_pfile,alpha))
                pareto_parchive = prune_archive(pareto_parchive,typePred =True, pref= pref)


                # Checking if current predictor point is dominated by initial pareto_archive  in the selected objectives
                # If dominated, ask for new preference and restart predictor loop
                if (pred > num_pred-2 and is_dominated(val_plosses, init_losses,pref)):
                    #print(f"❌ Current predictor validation point {val_plosses} is dominated by initial loss {init_losses}; restarting predictor loop.")
                    print(f"❌change too small, change preference/objectives/stepsize .......... Reverting to previous state")
                    pref = get_preference(nobj)
                    lrt = float(input(f"Enter new step size ") )
                    if count == 0:
                        prev_v= None
                        # Load back initial model
                        model.load_model(path / ckpt_file)
                    else:
                        prev_v= prev
                        model.load_model(path / ckpt_cfile)
                    pred = 0
                    backt = False
                    print(f"🔁 Restarting predictor loop with new p={pref}")
                    # continue → goes back to while pred < num_pred (now pred == 0)
                    continue

                elif count > 0 and pred > num_pred-2:
                    pred_old = []
                    pred_curr = []
                    for i in range(len(pref)):
                        if pref[i] < 0:
                            pred_old.append(corr_losses_final[i])
                            pred_curr.append(val_plosses[i])

                    if not np.all(np.array(pred_curr) <= np.array(pred_old)): #
                        print("No progress made, try increasing or reducing step size or changing preference")
                        print("current corrector",corr_losses_final)
                        pref = get_preference(nobj)
                        #backt = False
                        lrt = float(input(f"Enter new step size ") )
                        pred = 0
                        backt = False
                        print(f"🔁 Restarting predictor loop with new p={pref}")
                        # continue → goes back to while pred < num_pred (now pred == 0)
                        continue

                
                elif (count >= 0 and pred > 0): #num_pred-2 #0 # change number to increase steps sizes taken
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
                        print("Early stopping desired objectives have been minimized")
                        model.save_model(path/ckpt_pfile)
                        pred_losses = [v for v in pareto_parchive[-1][0]]
                        print(f"[Epoch {pred}] train/val:   {pred_losses}")
                        model.load_model(path / ckpt_pfile)
                        break

                
                elif (pred > num_pred-2 and any(is_dominated(val_plosses, prev_v,pref) for prev_v, _,_ in pareto_parchive if (prev_v != np.array(val_plosses)).any())):
                    #print("❌ Current predictor validation point is dominated by previous predictor archive; stopping. Objectives not optimized reverting back")
                    print("change too small, change preference/objectives .......... Reverting to previous state")
                    if count == 0:
                        # Load back initial model
                        model.load_model(path / ckpt_file)
                    else:
                        prev_v= prev
                        model.load_model(path / ckpt_cfile)
                    break

                elif pred == num_pred-1:
                    print("✅ Predictor loop finished all iterations without minimizing desired objectives or being dominated by initial point; stopping at last predictor.")
                    action, pref, max_runs = request_user_action(pref, num_pred)
                    if action == "change_pref":
                        # Restart predictor loop with new preference
                        pred = 0
                        backt = False
                        print(f"🔁 Restarting predictor loop with new p={pref}")
                        continue
                    elif action == "rerun_predictor":
                        # Update max_runs and restart predictor loop
                        num_pred = max_runs
                        pred = 0
                        backt = False
                        print(f"🔁 Restarting predictor loop with new max_runs={max_runs}")
                        continue

               

                # --- Save only if checkpoint is nondominated
                if any(np.array_equal(val_plosses, v) for v, _, _ in pareto_parchive):
                    model.save_model(path/ckpt_pfile)
                    pareto_parchive = pareto_parchive  # keep only nondominated points
                    model.load_model(path / ckpt_pfile)
                else:
                    # Current checkpoint is dominated, do not save
                    pareto_parchive = pareto_parchive 
                    model.load_model(path / ckpt_pfile)
              
                pred_losses = [v for v in pareto_parchive[-1][0]]
                print(f"[Epoch {pred}] train/val:   {pred_losses}")

            else:
                pref = get_preference(nobj)
                #lrt = float(input(f"Enter new step size ") )
                pred = 0
                backt = False
                print(f"🔁 Restarting predictor loop with new p={pref}")
                # continue → goes back to while pred < num_pred (now pred == 0)
                continue

            pred += 1

        torch.cuda.synchronize()
        train_pred_time = time.time() - t0_pred
        
        
        # --- Test evaluation for this nondominated checkpoint
        test_plosses, test_ptop1s = evaluate(model, testloader, nobj, criterion)
        #pred_losses = [v for v in pareto_parchive[-1][0]]
        eval_plosses = test_plosses
        pred_losses_old = pred_losses.copy()
        #print(f"[Epoch {pred}] {pred_losses}")
        print(f"[Epoch {pred}] train/val:   {pred_losses}")
        print(f"[Epoch {pred}] test losses: {test_plosses}, test acc: {test_ptop1s}")
        print("alpha_pred", alpha)

        

        pred_losses_final = pred_losses.copy()         
        print(f"Predictor time for preference {pref}: ", train_pred_time)
        # -------------- when predictor loop finishes successfully --------------
        print("✅ Predictor loop finished successfully. Proceeding to Corrector phase.")
            
        best_val_accuracy =  0.0
        total_closses = [0.0 for _ in range(num_tasks)]
        num_csamples = 0
        #best_val_loss=  float("inf") #0.0
        corr_losses_old = pred_losses.copy()
        eval_closses_old = eval_plosses.copy()
        pareto_carchive = []

        torch.cuda.synchronize()
        for corr in trange(num_corr):
            torch.cuda.synchronize()
            t0_corr = time.time()
            model.train()
            if  corr == 0:
                model.load_model(path / ckpt_pfile)
            
            else:
                model.load_model(path / ckpt_cfile)
            
            for batch_idx,(images, labels) in enumerate(tqdm(trainloader, desc='Batch', leave=False)):
                optimizer.zero_grad(set_to_none=True)
                images = images.to(device)
                labels = labels.to(device)
                
                alpha_corr = mgda_optimize(model=model,images=images,labels=labels,criterion=criterion,compute_alpha=compute_alpha,optimizer=optimizer_c,lr_scheduler=lr_scheduler_c)
                #alpha_corr = pmgda_optimize(model,images,labels,trainloader,criterion,pref, optimizer_c,lr_scheduler_c,init_lr=1e-2) 
                
                outs = model(images)  # list of m logits [B, C]
                for i in range(len(outs)):
                    total_closses[i] += criterion(outs[i], labels[:, i].long()).item() * images.size(0)
            
                num_csamples += images.size(0)

            corr_losses_ = [tl / num_csamples for tl in total_closses]
            print("Corrector train losses: ", corr_losses_)
            # --- Checkpoint path
            ckpt_cfile = f"model_corr_{num_corr}_{type}.pth"
            if corr == 0:
                #initialize model file
                model.save_model(path/ckpt_cfile)

             # ---- VALIDATION (model selection / early stopping) ----
            val_closses, _ = evaluate(model, valloader, nobj, criterion)
            #if lr_scheduler_c is not None:
            #    val_closs_scalar = sum(val_closses) / len(val_closses)
            #    lr_scheduler_c.step(val_closs_scalar)

            # --- Add to Pareto archive and prune dominated points
            pareto_carchive.append((np.array(val_closses), ckpt_cfile, alpha_corr))
            pruned_carchive = prune_archive(pareto_carchive)



            best_val_corr, best_ckpt_corr, best_alpha =  min(pareto_carchive,key=lambda x: np.mean(x[0])) ## find_min_mean(pruned_carchive, pref)
            if any(np.array_equal(val_closses, v) for v, _, _ in pruned_carchive):
                # Only save if it survives pruning (nondominated)
                #model.save_model(path/ckpt_cfile)
                save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                pareto_carchive = pruned_carchive  # keep only nondominated points
            else:
                # Current checkpoint is dominated, do not save
                pareto_carchive = pruned_carchive  # still update archive to remove dominated ones




            corr_losses1 = best_val_corr #[v for v in pareto_carchive[-1][0]]
            print(f"[Epoch {corr}] train/val:   {corr_losses1}")


            dominated_by_carchive = is_dominated(val_closses, init_losses)
            if (corr > num_corr-2 and is_dominated(val_closses, pred_losses, pref)):
                print("❌ Current corrector validation point is dominated by predictor point; stopping.")

                if is_dominated(best_val_corr, pred_losses, pref): # The predictor point is optimal. No change
                    print("✅ Best corrector validation point is dominated by predictor; stopping at predictor.")
                    corr_losses1 = pred_losses.copy()
                    alpha_corr = alpha
                    print(f"[Epoch {corr}] train/val: {corr_losses1}")
                    print("alpha_corr reset to alpha_pred: ", alpha_corr)
                    model.load_model(path / ckpt_pfile)
                    model.save_model(path / ckpt_cfile) 

                elif is_dominated(best_val, best_val_corr):
                    dm_d = dm_desire()
                    if dm_d:
                        save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                        model.load_model(path / ckpt_cfile)
                        corr_losses1 = best_val_corr
                        init_losses = best_val_corr # update initial losses to best corrector front for next predictor loop comparison
                        alpha_corr = best_alpha
                        print("alpha_corr ", alpha_corr)
                        print(f"[Epoch {corr}] train/val: {corr_losses1}")
                    else: 
                        print("❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.")
                        corr_losses1 = corr_losses_final.copy()#pred_losses.copy()
                        model.load_model(path / f"model_corr_{num_corr}_{type}_old.pth")
                        alpha_corr = alpha_corr_old 
                        print("alpha_corr reset to previous corrector alpha: ", alpha_corr)
                        print(f"[Epoch {corr}] train/val: {corr_losses1}")
                        model.save_model(path / ckpt_cfile)
                        cont = 1
                else:
                    #Converged, save current point
                    print("✅ Best corrector validation point is non-dominated; stopping at best corrector.")
                    save_or_copy(model,best_ckpt_corr, path/ckpt_cfile) #model.save_model(path/ckpt_cfile)#save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                    model.load_model(path / ckpt_cfile)
                    corr_losses1 = best_val_corr
                    alpha_corr = best_alpha
                    print("alpha_corr ", alpha_corr)
                    print(f"[Epoch {corr}] train/val: {corr_losses1}")
                    #print(f"[Epoch {corr}] test losses: {eval_closses}")
                break
 

            elif (corr > num_corr -2 and any(is_dominated(val_closses, prev_cv,pref) for prev_cv, _, _ in pareto_carchive if (prev_cv != np.array(val_closses)).any())):
                print("❌ Current corrector validation point is dominated by corrector archive; stopping at  previous/best corrector.")
                if is_dominated(best_val, best_val_corr): # A new front is obtained. Stick to the old front (discard)
                    dm_d = dm_desire()
                    if dm_d:
                        save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                        model.load_model(path / ckpt_cfile)
                        corr_losses1 = best_val_corr
                        init_losses = best_val_corr # update initial losses to best corrector front for next predictor loop comparison
                        alpha_corr = best_alpha
                        print("alpha_corr ", alpha_corr)
                        print(f"[Epoch {corr}] train/val: {corr_losses1}")
                    else: 
                        print("❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.")
                        corr_losses1 = corr_losses_final.copy()#pred_losses.copy()
                        model.load_model(path / f"model_corr_{num_corr}_{type}_old.pth")
                        alpha_corr = alpha_corr_old 
                        print("alpha_corr reset to previous corrector alpha: ", alpha_corr)
                        print(f"[Epoch {corr}] train/val: {corr_losses1}")
                        model.save_model(path / ckpt_cfile)
                        cont = 1

                elif is_dominated(best_val_corr, pred_losses): # The predictor point is optimal. No change
                    print("✅ Best corrector validation point is dominated by predictor; stopping at predictor.")
                    corr_losses1 = pred_losses.copy()
                    print(f"[Epoch {corr}] train/val: {corr_losses1}")
                    model.load_model(path / ckpt_pfile)
                    model.save_model(path / ckpt_cfile) 
                else:
                    #Converged, save current point
                    print("✅ Best corrector validation point is non-dominated; stopping at best corrector.")
                    save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                    model.load_model(path / ckpt_cfile)
                    corr_losses1 = best_val_corr
                    alpha_corr = best_alpha
                    print("alpha_corr ", alpha_corr)
                    print(f"[Epoch {corr}] train/val: {corr_losses1}")
                break

            elif (corr > num_corr- 2 and dominated_by_carchive):
                print("Current corrector validation point is dominated by initial point; stopping.")
                if is_dominated(best_val, best_val_corr): # A new front is obtained. Stick to the old front (discard)
                    dm_d = dm_desire()
                    if dm_d:
                        save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                        model.load_model(path / ckpt_cfile)
                        corr_losses1 = best_val_corr
                        init_losses = best_val_corr # update initial losses to best corrector front for next predictor loop comparison
                        alpha_corr = best_alpha
                        print("alpha_corr ", alpha_corr)
                        print(f"[Epoch {corr}] train/val: {corr_losses1}")
                    else: 
                        print("❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.")
                        corr_losses1 = corr_losses_final.copy()#pred_losses.copy()
                        model.load_model(path / f"model_corr_{num_corr}_{type}_old.pth")
                        alpha_corr = alpha_corr_old 
                        print("alpha_corr reset to previous corrector alpha: ", alpha_corr)
                        print(f"[Epoch {corr}] train/val: {corr_losses1}")
                        model.save_model(path / ckpt_cfile)
                        cont = 1

                elif is_dominated(best_val_corr, pred_losses, pref): # The predictor point is optimal. No change
                    print("✅ Best corrector validation point is dominated by predictor; stopping at predictor.")
                    corr_losses1 = pred_losses.copy()
                    alpha_corr = alpha
                    print("alpha_corr reset to alpha_pred: ", alpha_corr)
                    print(f"[Epoch {corr}] train/val: {corr_losses1}")
                    model.load_model(path / ckpt_pfile)
                    model.save_model(path / ckpt_cfile) 
                else:
                    #Converged, save current point
                    print("✅ Best corrector validation point is non-dominated; stopping at best corrector.")
                    save_or_copy(model,best_ckpt_corr, path/ckpt_cfile) #model.save_model(path/ckpt_cfile) #save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                    model.load_model(path / ckpt_cfile)
                    corr_losses1 = best_val_corr
                    alpha_corr = best_alpha
                    print("alpha_corr ", alpha_corr)
                    print(f"[Epoch {corr}] train/val: {corr_losses1}")
                break

            elif (count >= 0 and corr > 0): #0 # change number to increase steps sizes taken
                   
                if np.all(np.array(best_val_corr) <= np.array(pred_losses_final)):
                    print("Early stopping desired objectives have all been minimized")

                    if is_dominated(best_val, best_val_corr): # A new front is obtained. Stick to the old front (discard)
                        dm_d = dm_desire()
                        if dm_d:
                            save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                            model.load_model(path / ckpt_cfile)
                            corr_losses1 = best_val_corr
                            init_losses = best_val_corr # update initial losses to best corrector front for next predictor loop comparison
                            alpha_corr = best_alpha
                            print("alpha_corr ", alpha_corr)
                            print(f"[Epoch {corr}] train/val: {corr_losses1}")
                        else: 
                            print("❌ No desire to continue with new front. Reverting back to the last corrector. Take a different preference weights or objectives.")
                            corr_losses1 = corr_losses_final.copy()#pred_losses.copy()
                            model.load_model(path / f"model_corr_{num_corr}_{type}_old.pth")
                            alpha_corr = alpha_corr_old 
                            print("alpha_corr reset to previous corrector alpha: ", alpha_corr)
                            print(f"[Epoch {corr}] train/val: {corr_losses1}")
                            model.save_model(path / ckpt_cfile)
                            cont = 1
                    else:
                        #Converged, save current point
                        print("✅ Best corrector validation point is non-dominated; stopping at best corrector.")
                        save_or_copy(model,best_ckpt_corr, path/ckpt_cfile)
                        corr_losses1 = best_val_corr #[v for v in pareto_carchive[-1][0]]
                        alpha_corr = best_alpha
                        print("alpha_corr ", alpha_corr)
                        print(f"[Epoch {pred}] train/val:   {corr_losses1}")
                    break


        torch.cuda.synchronize()
        train_corr_time = time.time() - t0_corr
        if cont == 1: ## For reverting to previous corrector when new front is obtained but no desire to change
            continue

        model.load_model(path / ckpt_cfile)
        # --- Test evaluation for this nondominated checkpoint
        test_closses, test_ctop1s = evaluate(model, testloader, nobj, criterion)
        eval_closses = test_closses
        #print(f"[Epoch {corr}] train: {corr_losses1}")
        print(f"[Epoch {corr}] train/val:   {corr_losses1}")
        print(f"[Epoch {corr}] test losses: {test_closses}, test acc: {test_ctop1s}")


        if count == 0 and pred > 0:
            #ax.plot([init_loss1[-1,0], pred_loss1[0]], [init_loss2[-1,1], pred_loss2[1]], [init_loss3[-1,2], pred_loss3[2]],"-o", label='predictor step', color='blue',linewidth=2)
            ax.plot([init_loss[0][-1], pred_losses[0]], [init_loss[1][-1], pred_losses[1]], [init_loss[2][-1], pred_losses[2]],"-o", label='predictor step', color='blue',linewidth=2)

        elif count > 0 and pred > 0:
            ax.plot([pred_losses[0], corr_losses_final[0]], [pred_losses[1], corr_losses_final[1]], [pred_losses[2], corr_losses_final[2]],"-o", label='predictor step', color='blue',linewidth=2)
           
        else:
            print("NO Change")
        
        # Save all predictor and corrector losses for plotting trajectories
        for i in range(nobj):
            pred_loss_all[i].append(pred_losses[i])
            predt_loss_all[i].append(eval_plosses[i])

        for i in range(nobj):
            corr_loss_all[i].append(corr_losses1[i])
            corrt_loss_all[i].append(eval_closses[i])

        # Save the final corrector point for comparison in next iteration
        corr_losses_final = corr_losses1.copy()
        model.save_model(path / f"model_corr_{num_corr}_{type}_old.pth")
        alpha_corr_old = alpha_corr
        
        print(f"corrector time for preference {pref}: ", train_corr_time)
        #print("Ind-Corr test Accuracy", eval_ctop1s)
        #print("Corr test Accuracy", val_caccuracy)
        if not pred == 0:
            ax.plot([corr_loss_all[0][-1], pred_losses[0] ], [corr_loss_all[1][-1], pred_losses[1]], [corr_loss_all[2][-1], pred_losses[2]],"-o", label='corrector step', color='red',linewidth=2)

            # Clear previous plot and replot
            if type == "UCI":
                ax.set_xlabel("age-1", fontsize = 20)
                ax.set_ylabel("education-2", fontsize = 20)
                ax.set_zlabel("marital_stat-3",labelpad=10, fontsize = 20)
            
            elif type =="INT":
                ax.set_xlabel("fmnist-1", fontsize = 20)
                ax.set_ylabel("kmnist-2", fontsize = 20)
                ax.set_zlabel("mnist-3",labelpad=10, fontsize = 20)

            else:
                ax.set_xlabel("f1", fontsize = 20)
                ax.set_ylabel("f2", fontsize = 20)
                ax.set_zlabel("f3",labelpad=10, fontsize = 20)

            # Pause to allow plot to update
            plt.pause(0.5)

            for i in range(nobj):
                pred_loss[i].append(pred_loss_all[i][-1])
                corr_loss[i].append(corr_loss_all[i][-1])
                predt_loss[i].append(predt_loss_all[i][-1])
                corrt_loss[i].append(corrt_loss_all[i][-1])


            preference_all.append(preference[-1])
            alphas_all.append(alphat)
            alphas_all.append(alpha_corr)


            #initial_point = np.stack((init_loss[0],init_loss[1], init_loss[2]), axis=1)  
            initial_point =  np.array(init_loss).T
            predictor_point =  np.array(pred_loss).T
            corrector_point = np.array(corr_loss).T

            initialt_point = np.array(initt_loss).T
            predictort_point = np.array(predt_loss).T
            correctort_point = np.array(corrt_loss).T
            preference = np.array(preference)
            #alphas = np.array(alphas_all)
            alphas = np.array([a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)for a in alphas_all])

            total_init_loss_array = np.array(total_init_loss)
            total_time = train_pred_time + train_corr_time
            print(f"Total time for iteration {count}: ", total_time)


            try:
                # Works in normal Python scripts
                FILE_DIR = Path(__file__).resolve().parent
            except NameError:
                # Fallback for Jupyter/IPython
                FILE_DIR = Path(os.getcwd()).resolve()
            file_path = FILE_DIR.parent / f"Results/{type}"
            file_path.mkdir(parents=True, exist_ok=True)  # <-- ensure folder exists

            if count == 0: # Save once
                np.save(file_path /f"alpha_{num_init}.npy", alphat)

            with open(file_path / f'first_result_cen{count}.pkl', 'wb') as f:
                pickle.dump((initial_point,predictor_point,corrector_point,preference), f)

            with open(file_path / f'first_result_test_cen{count}.pkl', 'wb') as f:
                pickle.dump((initialt_point,predictort_point,correctort_point,preference), f)

            with open(file_path / f'first_alphas_cen{count}.pkl', 'wb') as f:
                pickle.dump((preference,alphas), f)

            with open(file_path / f'init_total_loss_cen{count}.pkl', 'wb') as f:
                pickle.dump(total_init_loss_array, f)

            with open(file_path / f'info_cen{count}.txt', "a") as f:
                f.write("=== Training ===\n")
                #f.write(f"Initial training time/ epoch {num_init}: {train_init_time:.2f} s\n\n")
                f.write(f"Predictor time / epoch {num_pred}: {train_pred_time:.2f} s\n\n")
                f.write(f"Corrector time / epoch {num_corr}: {train_corr_time:.2f} s\n\n")
                f.write(f"Training time (total): {total_time:.2f} s\n")

        
        count += 1
        print("COUNT: ", count)




